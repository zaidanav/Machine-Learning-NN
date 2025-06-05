import { Address } from '../core/Address';
import { Logger } from '../utils/Logger';
import { MembershipManager } from '../core/MembershipManager';
import { RaftNode, MembershipRequest, MembershipResponse } from '../core/RaftNode';
import { KVStore } from '../app/KVStore';
import { SimpleRpcClient } from '../rpc/SimpleRpcClient';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';

export interface NodeConfig {
  nodeId: string;
  ip: string;
  port: number;
  contactIp?: string;
  contactPort?: number;
  clusterInit?: boolean;
  dataDir?: string;
  logLevel?: string;
}

export interface ClusterConfig {
  nodes: NodeConfig[];
  networkConfig?: {
    enableSimulation: boolean;
    delay?: string;
    packetLoss?: string;
    corruption?: string;
  };
}

export interface NodeStatus {
  nodeId: string;
  address: string;
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'failed';
  role: 'leader' | 'follower' | 'candidate' | 'unknown';
  term: number;
  lastSeen: number;
  isHealthy: boolean;
  startTime: number;
  clusterSize: number;
}

export class InfrastructureManager extends EventEmitter {
  private logger: Logger;
  private nodeStatuses: Map<string, NodeStatus> = new Map();
  private managedNodes: Map<string, {
    raftNode: RaftNode;
    membershipManager: MembershipManager;
    kvStore: KVStore;
  }> = new Map();
  
  // Configuration
  private readonly STATUS_CHECK_INTERVAL = 5000; // 5 seconds
  private readonly HEALTH_CHECK_TIMEOUT = 3000; // 3 seconds
  private readonly NODE_START_TIMEOUT = 30000; // 30 seconds

  // Timers
  private statusCheckTimer: NodeJS.Timeout | null = null;

  constructor() {
    super();
    
    this.logger = new Logger({ 
      nodeId: 'infrastructure-manager'
    });

    this.logger.info('Infrastructure manager initialized');
  }

  /**
   * Initialize single node with configuration
   */
  async initializeNode(config: NodeConfig): Promise<RaftNode> {
    const address = new Address(config.ip, config.port);
    const nodeLogger = new Logger({ 
      nodeId: config.nodeId,
      address 
    });

    this.logger.info('Initializing node', {
      nodeId: config.nodeId,
      nodeAddress: address.toString(),
      isBootstrap: config.clusterInit
    });

    try {
      // Create data directory
      if (config.dataDir) {
        await this.ensureDataDirectory(config.dataDir);
      }

      // Initialize components
      const kvStore = new KVStore(nodeLogger);
      const rpcClient = new SimpleRpcClient();
      const raftNode = new RaftNode(address, kvStore, nodeLogger, rpcClient);
      const membershipManager = new MembershipManager(address, nodeLogger, rpcClient);

      // Integrate membership manager with raft node
      this.setupMembershipIntegration(raftNode, membershipManager);

      // Store managed components
      this.managedNodes.set(config.nodeId, {
        raftNode,
        membershipManager,
        kvStore
      });

      // Initialize node status
      this.nodeStatuses.set(config.nodeId, {
        nodeId: config.nodeId,
        address: address.toString(),
        status: 'starting',
        role: 'unknown',
        term: 0,
        lastSeen: Date.now(),
        isHealthy: false,
        startTime: Date.now(),
        clusterSize: 1
      });

      // Start the node
      const contactAddress = config.contactIp && config.contactPort 
        ? new Address(config.contactIp, config.contactPort)
        : undefined;

      await raftNode.start(contactAddress);

      // Update status
      const status = this.nodeStatuses.get(config.nodeId)!;
      status.status = 'running';
      status.isHealthy = true;

      this.logger.info('Node initialized successfully', {
        nodeId: config.nodeId,
        nodeAddress: address.toString(),
        role: raftNode.isLeader() ? 'leader' : 'follower'
      });

      this.emit('nodeInitialized', { nodeId: config.nodeId, address });
      return raftNode;

    } catch (error) {
      const status = this.nodeStatuses.get(config.nodeId);
      if (status) {
        status.status = 'failed';
        status.isHealthy = false;
      }

      this.logger.error('Failed to initialize node', error as Error, {
        nodeId: config.nodeId,
        nodeAddress: address.toString(),
        operation: 'initializeNode'
      });

      this.emit('nodeInitializationFailed', { 
        nodeId: config.nodeId, 
        address, 
        error: (error as Error).message 
      });

      throw error;
    }
  }

  /**
   * Setup integration between RaftNode and MembershipManager
   * FIXED: Proper interface compatibility
   */
  private setupMembershipIntegration(
    raftNode: RaftNode, 
    membershipManager: MembershipManager
  ): void {
    // Handle leadership changes
    raftNode.on('leadershipAcquired', (data) => {
      membershipManager.becomeLeader(data.term);
    });

    raftNode.on('becameFollower', (data) => {
      membershipManager.becomeFollower(data.term, data.leader);
    });

    // Handle membership requests in RaftNode with proper interface conversion
    const originalHandleMembership = raftNode.handleMembershipApplication.bind(raftNode);
    raftNode.handleMembershipApplication = async (request: MembershipRequest): Promise<MembershipResponse> => {
      try {
        // Delegate to membership manager
        const membershipResponse = await membershipManager.handleMembershipApplication(
          request.nodeAddr,
          request.term
        );

        // Convert MembershipChangeResponse to MembershipResponse
        const raftResponse: MembershipResponse = {
          status: membershipResponse.success ? 'success' : (membershipResponse.redirect ? 'redirected' : 'error'),
          leaderAddr: membershipResponse.leaderAddr,
          clusterAddrList: membershipResponse.clusterState,
          term: membershipResponse.term,
          message: membershipResponse.message
        };

        return raftResponse;

      } catch (error) {
        this.logger.error('Membership application failed', error as Error, {
          requesterAddress: `${request.nodeAddr.ip}:${request.nodeAddr.port}`,
          operation: 'handleMembershipApplication'
        });

        // Return error response in correct format
        return {
          status: 'error',
          term: membershipManager.getClusterInfo().term,
          message: `Error: ${(error as Error).message}`
        };
      }
    };

    // Integrate cluster addresses
    const originalGetClusterAddresses = raftNode.getClusterAddresses.bind(raftNode);
    raftNode.getClusterAddresses = () => {
      return membershipManager.getClusterAddresses();
    };

    this.logger.debug('Membership integration setup completed');
  }

  /**
   * Start multiple nodes as a cluster
   */
  async startCluster(config: ClusterConfig): Promise<Map<string, RaftNode>> {
    this.logger.info('Starting cluster', {
      nodeCount: config.nodes.length,
      networkSimulation: config.networkConfig?.enableSimulation
    });

    const nodes = new Map<string, RaftNode>();
    const startPromises: Promise<void>[] = [];

    // Start all nodes concurrently
    for (const nodeConfig of config.nodes) {
      const promise = this.initializeNode(nodeConfig)
        .then(node => {
          nodes.set(nodeConfig.nodeId, node);
        })
        .catch(error => {
          this.logger.error('Failed to start node in cluster', error as Error, {
            nodeId: nodeConfig.nodeId,
            operation: 'startCluster'
          });
          throw error;
        });
      
      startPromises.push(promise);
      
      // Stagger node starts to avoid conflicts
      await this.sleep(1000);
    }

    try {
      await Promise.all(startPromises);
      
      // Start status monitoring
      this.startStatusMonitoring();

      // Apply network simulation if configured
      if (config.networkConfig?.enableSimulation) {
        await this.applyNetworkSimulation(config.networkConfig);
      }

      this.logger.info('Cluster started successfully', {
        nodeCount: nodes.size,
        leaderNodes: Array.from(nodes.values()).filter(n => n.isLeader()).length
      });

      this.emit('clusterStarted', { 
        nodeCount: nodes.size,
        nodes: Array.from(nodes.keys())
      });

      return nodes;

    } catch (error) {
      this.logger.error('Failed to start cluster', error as Error, {
        operation: 'startCluster'
      });
      
      // Cleanup any started nodes
      await this.stopAllNodes();
      throw error;
    }
  }

  /**
   * Stop all managed nodes
   */
  async stopAllNodes(): Promise<void> {
    this.logger.info('Stopping all nodes', {
      nodeCount: this.managedNodes.size
    });

    this.stopStatusMonitoring();

    const stopPromises: Promise<void>[] = [];

    for (const [nodeId, components] of this.managedNodes) {
      const promise = this.stopNode(nodeId, components);
      stopPromises.push(promise);
    }

    await Promise.allSettled(stopPromises);
    
    this.managedNodes.clear();
    this.nodeStatuses.clear();

    this.logger.info('All nodes stopped');
    this.emit('allNodesStopped');
  }

  /**
   * Stop specific node
   */
  async stopNode(nodeId: string, components?: {
    raftNode: RaftNode;
    membershipManager: MembershipManager;
    kvStore: KVStore;
  }): Promise<void> {
    const nodeComponents = components || this.managedNodes.get(nodeId);
    
    if (!nodeComponents) {
      this.logger.warn('Node not found for stopping', { 
        nodeId,
        operation: 'stopNode'
      });
      return;
    }

    const status = this.nodeStatuses.get(nodeId);
    if (status) {
      status.status = 'stopping';
    }

    this.logger.info('Stopping node', { 
      nodeId,
      operation: 'stopNode'
    });

    try {
      // Stop components in order
      await nodeComponents.raftNode.stop();
      nodeComponents.membershipManager.cleanup();

      // Update status
      if (status) {
        status.status = 'stopped';
        status.isHealthy = false;
      }

      this.managedNodes.delete(nodeId);

      this.logger.info('Node stopped successfully', { 
        nodeId,
        operation: 'stopNode'
      });
      this.emit('nodeStopped', { nodeId });

    } catch (error) {
      this.logger.error('Error stopping node', error as Error, { 
        nodeId,
        operation: 'stopNode'
      });
      
      if (status) {
        status.status = 'failed';
      }

      throw error;
    }
  }

  /**
   * Add new node to existing cluster
   */
  async addNodeToCluster(
    nodeConfig: NodeConfig,
    leaderAddress: Address
  ): Promise<RaftNode> {
    this.logger.info('Adding node to cluster', {
      nodeId: nodeConfig.nodeId,
      nodeAddress: `${nodeConfig.ip}:${nodeConfig.port}`,
      leaderAddress: leaderAddress.toString(),
      operation: 'addNodeToCluster'
    });

    try {
      // Initialize the new node
      const newNodeConfig = {
        ...nodeConfig,
        contactIp: leaderAddress.ip,
        contactPort: leaderAddress.port,
        clusterInit: false
      };

      const raftNode = await this.initializeNode(newNodeConfig);

      // Wait for node to join cluster
      await this.waitForNodeToJoinCluster(nodeConfig.nodeId, 10000);

      this.logger.info('Node successfully added to cluster', {
        nodeId: nodeConfig.nodeId,
        clusterSize: this.getClusterSize(),
        operation: 'addNodeToCluster'
      });

      this.emit('nodeAddedToCluster', { 
        nodeId: nodeConfig.nodeId,
        clusterSize: this.getClusterSize()
      });

      return raftNode;

    } catch (error) {
      this.logger.error('Failed to add node to cluster', error as Error, {
        nodeId: nodeConfig.nodeId,
        operation: 'addNodeToCluster'
      });

      // Cleanup if failed
      await this.removeNodeFromCluster(nodeConfig.nodeId);
      throw error;
    }
  }

  /**
   * Remove node from cluster
   */
  async removeNodeFromCluster(nodeId: string): Promise<void> {
    const components = this.managedNodes.get(nodeId);
    
    if (!components) {
      this.logger.warn('Node not found for removal', { 
        nodeId,
        operation: 'removeNodeFromCluster'
      });
      return;
    }

    this.logger.info('Removing node from cluster', { 
      nodeId,
      operation: 'removeNodeFromCluster'
    });

    try {
      // If this is a leader, we need to find another leader first
      if (components.raftNode.isLeader()) {
        this.logger.warn('Removing leader node - cluster will need new election', {
          nodeId,
          operation: 'removeNodeFromCluster'
        });
      }

      // Stop the node gracefully
      await this.stopNode(nodeId, components);

      this.logger.info('Node removed from cluster', {
        nodeId,
        remainingNodes: this.managedNodes.size,
        operation: 'removeNodeFromCluster'
      });

      this.emit('nodeRemovedFromCluster', { 
        nodeId,
        clusterSize: this.getClusterSize()
      });

    } catch (error) {
      this.logger.error('Failed to remove node from cluster', error as Error, {
        nodeId,
        operation: 'removeNodeFromCluster'
      });
      throw error;
    }
  }

  /**
   * Wait for node to join cluster
   */
  private async waitForNodeToJoinCluster(
    nodeId: string,
    timeout: number = 30000
  ): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const status = this.nodeStatuses.get(nodeId);
      
      if (status && status.isHealthy && status.clusterSize > 1) {
        return; // Successfully joined
      }

      await this.sleep(1000);
    }

    throw new Error(`Node ${nodeId} failed to join cluster within ${timeout}ms`);
  }

  /**
   * Start status monitoring
   */
  private startStatusMonitoring(): void {
    if (this.statusCheckTimer) {
      return; // Already running
    }

    this.statusCheckTimer = setInterval(() => {
      this.performStatusCheck();
    }, this.STATUS_CHECK_INTERVAL);

    this.logger.debug('Started status monitoring');
  }

  /**
   * Stop status monitoring
   */
  private stopStatusMonitoring(): void {
    if (this.statusCheckTimer) {
      clearInterval(this.statusCheckTimer);
      this.statusCheckTimer = null;
    }

    this.logger.debug('Stopped status monitoring');
  }

  /**
   * Perform status check on all nodes
   */
  private async performStatusCheck(): Promise<void> {
    const statusPromises: Promise<void>[] = [];

    for (const [nodeId, components] of this.managedNodes) {
      statusPromises.push(this.checkNodeStatus(nodeId, components));
    }

    await Promise.allSettled(statusPromises);
  }

  /**
   * Check status of specific node
   */
  private async checkNodeStatus(
    nodeId: string,
    components: {
      raftNode: RaftNode;
      membershipManager: MembershipManager;
      kvStore: KVStore;
    }
  ): Promise<void> {
    const status = this.nodeStatuses.get(nodeId);
    if (!status) {
      return;
    }

    try {
      // Get cluster info from raft node
      const clusterInfo = components.raftNode.getClusterInfo();
      
      // Update status
      status.role = clusterInfo.type.toLowerCase() as any;
      status.term = clusterInfo.currentTerm;
      status.clusterSize = clusterInfo.clusterSize;
      status.lastSeen = Date.now();
      status.isHealthy = true;

      // Check if role changed
      this.emit('nodeStatusUpdated', { nodeId, status: { ...status } });

    } catch (error) {
      this.logger.warn('Failed to check node status', {
        nodeId,
        errorMessage: (error as Error).message,
        operation: 'checkNodeStatus'
      });

      status.isHealthy = false;
      status.lastSeen = Date.now();
    }
  }

  /**
   * Apply network simulation
   */
  private async applyNetworkSimulation(networkConfig: {
    delay?: string;
    packetLoss?: string;
    corruption?: string;
  }): Promise<void> {
    this.logger.info('Applying network simulation', {
      ...networkConfig,
      operation: 'applyNetworkSimulation'
    });

    // This would typically integrate with Docker/system network tools
    // For now, we'll just log the configuration
    
    const conditions = {
      delay: networkConfig.delay || '100ms 50ms',
      loss: networkConfig.packetLoss || '2%',
      corruption: networkConfig.corruption || '1%'
    };

    this.logger.info('Network simulation applied', {
      ...conditions,
      operation: 'applyNetworkSimulation'
    });
    this.emit('networkSimulationApplied', conditions);
  }

  /**
   * Ensure data directory exists
   */
  private async ensureDataDirectory(dataDir: string): Promise<void> {
    try {
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
        this.logger.debug('Created data directory', { 
          dataDir,
          operation: 'ensureDataDirectory'
        });
      }
    } catch (error) {
      this.logger.error('Failed to create data directory', error as Error, {
        dataDir,
        operation: 'ensureDataDirectory'
      });
      throw error;
    }
  }

  /**
   * Get cluster status overview
   */
  getClusterStatus(): {
    totalNodes: number;
    healthyNodes: number;
    leaders: number;
    followers: number;
    candidates: number;
    failed: number;
    nodes: NodeStatus[];
  } {
    const nodes = Array.from(this.nodeStatuses.values());
    
    return {
      totalNodes: nodes.length,
      healthyNodes: nodes.filter(n => n.isHealthy).length,
      leaders: nodes.filter(n => n.role === 'leader').length,
      followers: nodes.filter(n => n.role === 'follower').length,
      candidates: nodes.filter(n => n.role === 'candidate').length,
      failed: nodes.filter(n => !n.isHealthy).length,
      nodes: [...nodes]
    };
  }

  /**
   * Get current cluster size
   */
  getClusterSize(): number {
    return this.managedNodes.size;
  }

  /**
   * Get managed node by ID
   */
  getNode(nodeId: string): RaftNode | undefined {
    return this.managedNodes.get(nodeId)?.raftNode;
  }

  /**
   * Get all managed nodes
   */
  getAllNodes(): Map<string, RaftNode> {
    const result = new Map<string, RaftNode>();
    for (const [nodeId, components] of this.managedNodes) {
      result.set(nodeId, components.raftNode);
    }
    return result;
  }

  /**
   * Find current leader node
   */
  findLeaderNode(): { nodeId: string; raftNode: RaftNode } | null {
    for (const [nodeId, components] of this.managedNodes) {
      if (components.raftNode.isLeader()) {
        return { nodeId, raftNode: components.raftNode };
      }
    }
    return null;
  }

  /**
   * Generate cluster configuration for deployment
   */
  generateClusterConfig(
    nodeCount: number,
    baseIp: string = '172.20.0',
    basePort: number = 8001
  ): ClusterConfig {
    const nodes: NodeConfig[] = [];

    for (let i = 0; i < nodeCount; i++) {
      const nodeConfig: NodeConfig = {
        nodeId: `node-${i + 1}`,
        ip: `${baseIp}.${10 + i}`,
        port: basePort,
        clusterInit: i === 0, // First node initializes cluster
        dataDir: `/app/data/node-${i + 1}`,
        logLevel: 'info'
      };

      // Set contact info for joining nodes
      if (i > 0) {
        nodeConfig.contactIp = `${baseIp}.10`;
        nodeConfig.contactPort = basePort;
      }

      nodes.push(nodeConfig);
    }

    return {
      nodes,
      networkConfig: {
        enableSimulation: false,
        delay: '100ms 50ms',
        packetLoss: '2%',
        corruption: '1%'
      }
    };
  }

  /**
   * Export cluster configuration to file
   */
  async exportClusterConfig(
    config: ClusterConfig,
    filePath: string
  ): Promise<void> {
    try {
      const configJson = JSON.stringify(config, null, 2);
      fs.writeFileSync(filePath, configJson);
      
      this.logger.info('Cluster configuration exported', { 
        filePath,
        operation: 'exportClusterConfig'
      });
    } catch (error) {
      this.logger.error('Failed to export cluster configuration', error as Error, {
        filePath,
        operation: 'exportClusterConfig'
      });
      throw error;
    }
  }

  /**
   * Import cluster configuration from file
   */
  async importClusterConfig(filePath: string): Promise<ClusterConfig> {
    try {
      const configJson = fs.readFileSync(filePath, 'utf-8');
      const config = JSON.parse(configJson) as ClusterConfig;
      
      this.logger.info('Cluster configuration imported', { 
        filePath,
        operation: 'importClusterConfig'
      });
      return config;
    } catch (error) {
      this.logger.error('Failed to import cluster configuration', error as Error, {
        filePath,
        operation: 'importClusterConfig'
      });
      throw error;
    }
  }

  /**
   * Utility sleep function
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Cleanup all resources
   */
  async cleanup(): Promise<void> {
    this.logger.info('Cleaning up infrastructure manager');
    
    await this.stopAllNodes();
    this.removeAllListeners();
    
    this.logger.info('Infrastructure manager cleanup completed');
  }
}