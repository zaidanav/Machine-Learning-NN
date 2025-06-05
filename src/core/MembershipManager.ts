import { Address, AddressData } from './Address';
import { Logger } from '../utils/Logger';
import { EventEmitter } from 'events';

export interface NodeInfo {
  address: Address;
  nodeId: string;
  joinTime: number;
  lastSeen: number;
  status: 'active' | 'suspected' | 'failed' | 'leaving';
  role: 'leader' | 'follower' | 'candidate';
  term: number;
}

export interface MembershipChangeRequest {
  type: 'add' | 'remove';
  nodeAddr: AddressData;
  requesterId: AddressData;
  term: number;
  timestamp: number;
}

export interface MembershipChangeResponse {
  success: boolean;
  clusterState: AddressData[];
  leaderAddr?: AddressData;
  term: number;
  message?: string;
  redirect?: boolean;
}

export interface ClusterState {
  members: Map<string, NodeInfo>;
  leader: Address | null;
  term: number;
  lastUpdate: number;
  size: number;
}

export class MembershipManager extends EventEmitter {
  private logger: Logger;
  private rpcClient: any;
  private clusterState: ClusterState;
  private selfAddress: Address;
  private isLeader: boolean = false;
  
  // Configuration
  private readonly FAILURE_DETECTION_TIMEOUT = 10000; // 10 seconds
  private readonly MEMBERSHIP_SYNC_INTERVAL = 5000; // 5 seconds
  private readonly MAX_RETRY_ATTEMPTS = 3;

  // Timers
  private failureDetectionTimer: NodeJS.Timeout | null = null;
  private membershipSyncTimer: NodeJS.Timeout | null = null;

  constructor(
    selfAddress: Address,
    logger: Logger,
    rpcClient: any
  ) {
    super();
    
    this.selfAddress = selfAddress;
    this.logger = logger;
    this.rpcClient = rpcClient;
    
    this.clusterState = {
      members: new Map(),
      leader: null,
      term: 0,
      lastUpdate: Date.now(),
      size: 0
    };

    // Add self to cluster
    this.addMember(selfAddress, process.env.NODE_ID || 'unknown', 'follower', 0);

    this.logger.membership('Membership manager initialized', {
      selfAddress: selfAddress.toString(),
      initialClusterSize: 1
    });
  }

  /**
   * Initialize cluster as bootstrap leader
   */
  initializeAsLeader(term: number): void {
    this.isLeader = true;
    this.clusterState.leader = this.selfAddress;
    this.clusterState.term = term;
    this.clusterState.lastUpdate = Date.now();

    // Update self info
    const selfInfo = this.clusterState.members.get(this.selfAddress.hashCode());
    if (selfInfo) {
      selfInfo.role = 'leader';
      selfInfo.term = term;
      selfInfo.lastSeen = Date.now();
    }

    this.startPeriodicTasks();

    this.logger.membership('Initialized as bootstrap leader', {
      term,
      clusterSize: this.clusterState.size
    });

    this.emit('leadershipAcquired', { term, clusterSize: this.clusterState.size });
  }

  /**
   * Handle leadership acquisition
   */
  becomeLeader(term: number): void {
    this.isLeader = true;
    this.clusterState.leader = this.selfAddress;
    this.clusterState.term = term;
    this.clusterState.lastUpdate = Date.now();

    // Update self info
    const selfInfo = this.clusterState.members.get(this.selfAddress.hashCode());
    if (selfInfo) {
      selfInfo.role = 'leader';
      selfInfo.term = term;
      selfInfo.lastSeen = Date.now();
    }

    this.startPeriodicTasks();

    this.logger.membership('Became cluster leader', {
      term,
      clusterSize: this.clusterState.size,
      members: Array.from(this.clusterState.members.keys())
    });

    this.emit('leadershipAcquired', { term, clusterSize: this.clusterState.size });
  }

  /**
   * Handle leadership loss
   */
  becomeFollower(term: number, newLeader?: Address): void {
    this.isLeader = false;
    this.clusterState.leader = newLeader || null;
    this.clusterState.term = term;
    this.clusterState.lastUpdate = Date.now();

    // Update self info
    const selfInfo = this.clusterState.members.get(this.selfAddress.hashCode());
    if (selfInfo) {
      selfInfo.role = 'follower';
      selfInfo.term = term;
      selfInfo.lastSeen = Date.now();
    }

    this.stopPeriodicTasks();

    this.logger.membership('Became follower', {
      term,
      newLeader: newLeader?.toString(),
      clusterSize: this.clusterState.size
    });

    this.emit('becameFollower', { term, leader: newLeader });
  }

  /**
   * Handle membership application from new node
   */
  async handleMembershipApplication(
    nodeAddr: AddressData, 
    requestTerm: number
  ): Promise<MembershipChangeResponse> {
    const requesterAddr = Address.fromObject(nodeAddr);
    
    this.logger.membership('Received membership application', {
      requesterAddress: requesterAddr.toString(),
      requestTerm,
      currentTerm: this.clusterState.term,
      isLeader: this.isLeader,
      currentClusterSize: this.clusterState.size
    });

    // Check if we're the leader
    if (!this.isLeader) {
      return {
        success: false,
        clusterState: [],
        leaderAddr: this.clusterState.leader?.toObject(),
        term: this.clusterState.term,
        message: 'Not the leader',
        redirect: true
      };
    }

    // Check term
    if (requestTerm > this.clusterState.term) {
      this.logger.membership('Higher term detected, stepping down', {
        requestTerm,
        currentTerm: this.clusterState.term
      });
      
      this.emit('higherTermDetected', { term: requestTerm });
      
      return {
        success: false,
        clusterState: [],
        term: requestTerm,
        message: 'Higher term detected, leadership lost'
      };
    }

    try {
      // Add node to cluster
      const success = await this.addNodeToCluster(requesterAddr);
      
      if (success) {
        const clusterAddresses = Array.from(this.clusterState.members.values())
          .map(member => member.address.toObject());

        this.logger.membership('Membership application approved', {
          newMemberAddress: requesterAddr.toString(),
          newClusterSize: this.clusterState.size,
          allMembers: clusterAddresses.map(addr => `${addr.ip}:${addr.port}`)
        });

        // Notify other nodes about new member
        this.broadcastMembershipChange('add', requesterAddr);

        return {
          success: true,
          clusterState: clusterAddresses,
          leaderAddr: this.selfAddress.toObject(),
          term: this.clusterState.term,
          message: 'Successfully joined cluster'
        };
      } else {
        return {
          success: false,
          clusterState: [],
          term: this.clusterState.term,
          message: 'Failed to add node to cluster'
        };
      }

    } catch (error) {
      this.logger.error('Error processing membership application', error as Error, {
        requesterAddress: requesterAddr.toString(),
        operation: 'handleMembershipApplication'
      });

      return {
        success: false,
        clusterState: [],
        term: this.clusterState.term,
        message: `Error: ${(error as Error).message}`
      };
    }
  }

  /**
   * Add node to cluster
   */
  private async addNodeToCluster(nodeAddr: Address): Promise<boolean> {
    const nodeKey = nodeAddr.hashCode();
    
    // Check if node already exists
    if (this.clusterState.members.has(nodeKey)) {
      this.logger.membership('Node already in cluster', {
        nodeAddress: nodeAddr.toString()
      });
      return true;
    }

    try {
      // Test connectivity to new node
      const isReachable = await this.testNodeConnectivity(nodeAddr);
      if (!isReachable) {
        this.logger.warn('Cannot reach new node', {
          nodeAddress: nodeAddr.toString(),
          operation: 'addNodeToCluster'
        });
        return false;
      }

      // Add to cluster state
      this.addMember(
        nodeAddr, 
        `node-${nodeAddr.hashCode()}`, 
        'follower', 
        this.clusterState.term
      );

      this.logger.membership('Node successfully added to cluster', {
        nodeAddress: nodeAddr.toString(),
        clusterSize: this.clusterState.size
      });

      this.emit('memberAdded', { address: nodeAddr, clusterSize: this.clusterState.size });
      return true;

    } catch (error) {
      this.logger.error('Failed to add node to cluster', error as Error, {
        nodeAddress: nodeAddr.toString(),
        operation: 'addNodeToCluster'
      });
      return false;
    }
  }

  /**
   * Remove node from cluster
   */
  async removeNodeFromCluster(nodeAddr: Address): Promise<boolean> {
    if (!this.isLeader) {
      this.logger.warn('Cannot remove node: not the leader', {
        nodeAddress: nodeAddr.toString()
      });
      return false;
    }

    const nodeKey = nodeAddr.hashCode();
    const nodeInfo = this.clusterState.members.get(nodeKey);

    if (!nodeInfo) {
      this.logger.warn('Cannot remove node: not in cluster', {
        nodeAddress: nodeAddr.toString()
      });
      return false;
    }

    // Cannot remove leader (self)
    if (nodeAddr.equals(this.selfAddress)) {
      this.logger.warn('Cannot remove self from cluster', {
        nodeAddress: nodeAddr.toString()
      });
      return false;
    }

    try {
      // Mark as leaving
      nodeInfo.status = 'leaving';
      
      // Notify the node about removal (best effort)
      try {
        await this.rpcClient.call(
          nodeAddr.toURL(),
          'notifyRemoval',
          { 
            reason: 'manual_removal',
            term: this.clusterState.term 
          },
          { timeout: 2000, retries: 1 }
        );
      } catch (error) {
        // Ignore notification failure
        this.logger.warn('Could not notify node about removal', {
          nodeAddress: nodeAddr.toString(),
          errorMessage: (error as Error).message
        });
      }

      // Remove from cluster
      this.clusterState.members.delete(nodeKey);
      this.clusterState.size = this.clusterState.members.size;
      this.clusterState.lastUpdate = Date.now();

      this.logger.membership('Node removed from cluster', {
        nodeAddress: nodeAddr.toString(),
        clusterSize: this.clusterState.size
      });

      // Broadcast removal to other nodes
      this.broadcastMembershipChange('remove', nodeAddr);

      this.emit('memberRemoved', { address: nodeAddr, clusterSize: this.clusterState.size });
      return true;

    } catch (error) {
      this.logger.error('Failed to remove node from cluster', error as Error, {
        nodeAddress: nodeAddr.toString(),
        operation: 'removeNodeFromCluster'
      });
      return false;
    }
  }

  /**
   * Test connectivity to a node
   */
  private async testNodeConnectivity(nodeAddr: Address): Promise<boolean> {
    try {
      const response = await this.rpcClient.call(
        nodeAddr.toURL(),
        'ping',
        {},
        { timeout: 3000, retries: 1 }
      );
      return response.pong === true;
    } catch (error) {
      this.logger.debug('Node connectivity test failed', {
        nodeAddress: nodeAddr.toString(),
        errorMessage: (error as Error).message
      });
      return false;
    }
  }

  /**
   * Broadcast membership change to all nodes except the target
   */
  private async broadcastMembershipChange(
    changeType: 'add' | 'remove',
    targetAddr: Address
  ): Promise<void> {
    const membershipUpdate = {
      type: changeType,
      nodeAddr: targetAddr.toObject(),
      clusterState: Array.from(this.clusterState.members.values())
        .map(member => member.address.toObject()),
      term: this.clusterState.term,
      timestamp: Date.now()
    };

    const broadcastPromises: Promise<void>[] = [];

    for (const member of this.clusterState.members.values()) {
      // Skip self and target node
      if (member.address.equals(this.selfAddress) || 
          member.address.equals(targetAddr)) {
        continue;
      }

      broadcastPromises.push(
        this.notifyMembershipChange(member.address, membershipUpdate)
      );
    }

    // Send notifications concurrently (best effort)
    const results = await Promise.allSettled(broadcastPromises);
    
    const successCount = results.filter(r => r.status === 'fulfilled').length;
    const failureCount = results.filter(r => r.status === 'rejected').length;

    this.logger.membership('Membership change broadcast completed', {
      changeType,
      targetNodeAddress: targetAddr.toString(),
      successCount,
      failureCount,
      totalNotifications: broadcastPromises.length
    });
  }

  /**
   * Notify individual node about membership change
   */
  private async notifyMembershipChange(
    nodeAddr: Address,
    updateInfo: any
  ): Promise<void> {
    try {
      await this.rpcClient.call(
        nodeAddr.toURL(),
        'membershipUpdate',
        updateInfo,
        { timeout: 2000, retries: 1 }
      );

      this.logger.debug('Membership change notification sent', {
        targetNodeAddress: nodeAddr.toString()
      });

    } catch (error) {
      this.logger.warn('Failed to notify node about membership change', {
        nodeAddress: nodeAddr.toString(),
        errorMessage: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * Handle membership update from leader
   */
  handleMembershipUpdate(updateInfo: any): void {
    if (!this.isLeader && updateInfo.term >= this.clusterState.term) {
      this.logger.membership('Received membership update from leader', {
        changeType: updateInfo.type,
        targetNodeAddress: `${updateInfo.nodeAddr.ip}:${updateInfo.nodeAddr.port}`,
        newClusterState: updateInfo.clusterState?.length || 0
      });

      // Update local cluster state
      if (updateInfo.clusterState && Array.isArray(updateInfo.clusterState)) {
        this.syncClusterState(updateInfo.clusterState, updateInfo.term);
      }

      this.emit('membershipUpdated', updateInfo);
    }
  }

  /**
   * Sync cluster state from leader
   */
  syncClusterState(clusterAddresses: AddressData[], term: number): void {
    const oldSize = this.clusterState.size;
    
    // Clear and rebuild cluster state
    this.clusterState.members.clear();
    
    clusterAddresses.forEach((addrData, index) => {
      const addr = Address.fromObject(addrData);
      this.addMember(
        addr,
        `node-${addr.hashCode()}`,
        addr.equals(this.clusterState.leader) ? 'leader' : 'follower',
        term
      );
    });

    this.clusterState.term = term;
    this.clusterState.lastUpdate = Date.now();

    this.logger.membership('Cluster state synchronized', {
      oldSize,
      newSize: this.clusterState.size,
      term,
      members: Array.from(this.clusterState.members.keys())
    });
  }

  /**
   * Add member to cluster state
   */
  private addMember(
    address: Address,
    nodeId: string,
    role: 'leader' | 'follower' | 'candidate',
    term: number
  ): void {
    const nodeInfo: NodeInfo = {
      address,
      nodeId,
      joinTime: Date.now(),
      lastSeen: Date.now(),
      status: 'active',
      role,
      term
    };

    this.clusterState.members.set(address.hashCode(), nodeInfo);
    this.clusterState.size = this.clusterState.members.size;
  }

  /**
   * Start periodic tasks (leader only)
   */
  private startPeriodicTasks(): void {
    if (!this.isLeader) return;

    // Failure detection
    this.failureDetectionTimer = setInterval(() => {
      this.performFailureDetection();
    }, this.FAILURE_DETECTION_TIMEOUT);

    // Membership sync
    this.membershipSyncTimer = setInterval(() => {
      this.performMembershipSync();
    }, this.MEMBERSHIP_SYNC_INTERVAL);

    this.logger.debug('Started periodic membership tasks');
  }

  /**
   * Stop periodic tasks
   */
  private stopPeriodicTasks(): void {
    if (this.failureDetectionTimer) {
      clearInterval(this.failureDetectionTimer);
      this.failureDetectionTimer = null;
    }

    if (this.membershipSyncTimer) {
      clearInterval(this.membershipSyncTimer);
      this.membershipSyncTimer = null;
    }

    this.logger.debug('Stopped periodic membership tasks');
  }

  /**
   * Perform failure detection (leader only)
   */
  private async performFailureDetection(): Promise<void> {
    if (!this.isLeader) return;

    const now = Date.now();
    const suspectedNodes: NodeInfo[] = [];

    for (const member of this.clusterState.members.values()) {
      // Skip self
      if (member.address.equals(this.selfAddress)) {
        continue;
      }

      const timeSinceLastSeen = now - member.lastSeen;
      
      if (timeSinceLastSeen > this.FAILURE_DETECTION_TIMEOUT && 
          member.status === 'active') {
        
        member.status = 'suspected';
        suspectedNodes.push(member);
        
        this.logger.warn('Node suspected of failure', {
          nodeAddress: member.address.toString(),
          timeSinceLastSeen,
          threshold: this.FAILURE_DETECTION_TIMEOUT
        });
      }
    }

    // Verify suspected nodes
    for (const nodeInfo of suspectedNodes) {
      const isAlive = await this.testNodeConnectivity(nodeInfo.address);
      
      if (isAlive) {
        nodeInfo.status = 'active';
        nodeInfo.lastSeen = now;
        this.logger.info('Suspected node is actually alive', {
          nodeAddress: nodeInfo.address.toString()
        });
      } else {
        nodeInfo.status = 'failed';
        this.logger.error('Node confirmed failed', undefined, {
          nodeAddress: nodeInfo.address.toString()
        });
        
        this.emit('memberFailed', { address: nodeInfo.address });
        
        // Note: Auto-removal is disabled per specification
        // Manual removal required
      }
    }
  }

  /**
   * Perform membership sync (leader only)
   */
  private async performMembershipSync(): Promise<void> {
    if (!this.isLeader) return;

    // Update last seen for active members via heartbeat responses
    // This will be called by RaftNode when heartbeat responses are received
  }

  /**
   * Update member last seen time
   */
  updateMemberLastSeen(memberAddr: Address): void {
    const nodeKey = memberAddr.hashCode();
    const member = this.clusterState.members.get(nodeKey);
    
    if (member) {
      member.lastSeen = Date.now();
      if (member.status === 'suspected') {
        member.status = 'active';
        this.logger.info('Suspected member is active again', {
          nodeAddress: memberAddr.toString()
        });
      }
    }
  }

  /**
   * Get current cluster information
   */
  getClusterInfo(): {
    size: number;
    members: AddressData[];
    leader: AddressData | null;
    term: number;
    self: AddressData;
    memberDetails: Array<{
      address: string;
      nodeId: string;
      status: string;
      role: string;
      lastSeen: number;
    }>;
  } {
    return {
      size: this.clusterState.size,
      members: Array.from(this.clusterState.members.values())
        .map(member => member.address.toObject()),
      leader: this.clusterState.leader?.toObject() || null,
      term: this.clusterState.term,
      self: this.selfAddress.toObject(),
      memberDetails: Array.from(this.clusterState.members.values()).map(member => ({
        address: member.address.toString(),
        nodeId: member.nodeId,
        status: member.status,
        role: member.role,
        lastSeen: member.lastSeen
      }))
    };
  }

  /**
   * Get cluster addresses for Raft operations
   */
  getClusterAddresses(): Address[] {
    return Array.from(this.clusterState.members.values())
      .map(member => member.address);
  }

  /**
   * Check if node is in cluster
   */
  isMember(nodeAddr: Address): boolean {
    return this.clusterState.members.has(nodeAddr.hashCode());
  }

  /**
   * Get leader address
   */
  getLeaderAddress(): Address | null {
    return this.clusterState.leader;
  }

  /**
   * Check if membership manager is healthy
   */
  isHealthy(): boolean {
    return this.clusterState.size > 0;
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.stopPeriodicTasks();
    this.removeAllListeners();
    
    this.logger.membership('Membership manager cleaned up');
  }
}