import { Address, AddressData } from '../core/Address';
import { Logger } from '../utils/Logger';
import { MembershipManager } from '../core/MembershipManager';
import { RaftNode, MembershipRequest, MembershipResponse } from '../core/RaftNode';

export interface MembershipRpcHandler {
  applyMembership(request: MembershipRequest): Promise<MembershipResponse>;
  membershipUpdate(updateInfo: any): Promise<{ success: boolean; message?: string }>;
  notifyRemoval(info: { reason: string; term: number }): Promise<{ acknowledged: boolean }>;
  getClusterInfo(): Promise<{
    members: AddressData[];
    leader: AddressData | null;
    term: number;
    size: number;
  }>;
}

export class MembershipIntegration implements MembershipRpcHandler {
  private raftNode: RaftNode;
  private membershipManager: MembershipManager;
  private logger: Logger;

  constructor(
    raftNode: RaftNode,
    membershipManager: MembershipManager,
    logger: Logger
  ) {
    this.raftNode = raftNode;
    this.membershipManager = membershipManager;
    this.logger = logger;

    this.setupIntegration();
  }

  /**
   * Setup integration between RaftNode and MembershipManager
   */
  private setupIntegration(): void {
    // Handle leadership changes
    this.raftNode.on('leadershipAcquired', (data) => {
      this.membershipManager.becomeLeader(data.term);
      this.logger.membership('Membership manager notified of leadership acquisition', {
        term: data.term
      });
    });

    this.raftNode.on('becameFollower', (data) => {
      this.membershipManager.becomeFollower(data.term, data.leader);
      this.logger.membership('Membership manager notified of becoming follower', {
        term: data.term,
        leader: data.leader?.toString()
      });
    });

    // Handle membership manager events
    this.membershipManager.on('memberAdded', (data) => {
      this.logger.membership('Member added to cluster', {
        address: data.address.toString(),
        clusterSize: data.clusterSize
      });
    });

    this.membershipManager.on('memberRemoved', (data) => {
      this.logger.membership('Member removed from cluster', {
        address: data.address.toString(),
        clusterSize: data.clusterSize
      });
    });

    this.membershipManager.on('memberFailed', (data) => {
      this.logger.membership('Member failed', {
        address: data.address.toString()
      });
    });

    // Override RaftNode's membership handling
    this.overrideRaftNodeMembershipHandling();

    this.logger.membership('Membership integration setup completed');
  }

  /**
   * Override RaftNode's membership handling to use MembershipManager
   */
  private overrideRaftNodeMembershipHandling(): void {
    // Store original method
    const originalHandleMembership = this.raftNode.handleMembershipApplication.bind(this.raftNode);

    // Override with integration
    this.raftNode.handleMembershipApplication = async (request: MembershipRequest): Promise<MembershipResponse> => {
      return await this.applyMembership(request);
    };

    // Override cluster addresses method
    const originalGetClusterAddresses = this.raftNode.getClusterAddresses.bind(this.raftNode);
    this.raftNode.getClusterAddresses = (): Address[] => {
      return this.membershipManager.getClusterAddresses();
    };

    // Override leader address method
    const originalGetLeaderAddress = this.raftNode.getLeaderAddress.bind(this.raftNode);
    this.raftNode.getLeaderAddress = (): Address | null => {
      return this.membershipManager.getLeaderAddress() || originalGetLeaderAddress();
    };

    this.logger.debug('RaftNode membership methods overridden');
  }

  /**
   * Handle membership application RPC
   */
  async applyMembership(request: MembershipRequest): Promise<MembershipResponse> {
    this.logger.membership('Processing membership application', {
      nodeAddr: `${request.nodeAddr.ip}:${request.nodeAddr.port}`,
      term: request.term
    });

    try {
      const response = await this.membershipManager.handleMembershipApplication(
        request.nodeAddr,
        request.term
      );

      // Convert to RaftNode response format
      const raftResponse: MembershipResponse = {
        status: response.success ? 'success' : (response.redirect ? 'redirected' : 'error'),
        leaderAddr: response.leaderAddr,
        clusterAddrList: response.clusterState,
        term: response.term,
        message: response.message
      };

      this.logger.membership('Membership application processed', {
        nodeAddr: `${request.nodeAddr.ip}:${request.nodeAddr.port}`,
        status: raftResponse.status,
        clusterSize: response.clusterState?.length || 0
      });

      return raftResponse;

    } catch (error) {
      this.logger.error('Failed to process membership application', error as Error, {
        nodeAddr: `${request.nodeAddr.ip}:${request.nodeAddr.port}`
      });

      return {
        status: 'error',
        term: this.membershipManager.getClusterInfo().term,
        message: `Error: ${(error as Error).message}`
      };
    }
  }

  /**
   * Handle membership update RPC
   */
  async membershipUpdate(updateInfo: {
    type: 'add' | 'remove';
    nodeAddr: AddressData;
    clusterState?: AddressData[];
    term: number;
    timestamp: number;
  }): Promise<{ success: boolean; message?: string }> {
    this.logger.membership('Received membership update', {
      type: updateInfo.type,
      nodeAddr: `${updateInfo.nodeAddr.ip}:${updateInfo.nodeAddr.port}`,
      term: updateInfo.term,
      clusterStateSize: updateInfo.clusterState?.length
    });

    try {
      this.membershipManager.handleMembershipUpdate(updateInfo);

      return {
        success: true,
        message: 'Membership update applied'
      };

    } catch (error) {
      this.logger.error('Failed to apply membership update', error as Error);

      return {
        success: false,
        message: (error as Error).message
      };
    }
  }

  /**
   * Handle removal notification RPC
   */
  async notifyRemoval(info: {
    reason: string;
    term: number;
  }): Promise<{ acknowledged: boolean }> {
    this.logger.membership('Received removal notification', {
      reason: info.reason,
      term: info.term
    });

    try {
      // Gracefully shutdown
      setTimeout(async () => {
        await this.raftNode.stop();
        process.exit(0);
      }, 5000); // Give 5 seconds to acknowledge

      return {
        acknowledged: true
      };

    } catch (error) {
      this.logger.error('Failed to acknowledge removal', error as Error);

      return {
        acknowledged: false
      };
    }
  }

  /**
   * Get cluster information RPC
   */
  async getClusterInfo(): Promise<{
    members: AddressData[];
    leader: AddressData | null;
    term: number;
    size: number;
  }> {
    const clusterInfo = this.membershipManager.getClusterInfo();

    return {
      members: clusterInfo.members,
      leader: clusterInfo.leader,
      term: clusterInfo.term,
      size: clusterInfo.size
    };
  }

  /**
   * Manually add node to cluster (admin operation)
   */
  async addNode(nodeAddress: Address): Promise<{ success: boolean; message: string }> {
    if (!this.raftNode.isLeader()) {
      return {
        success: false,
        message: 'Only leader can add nodes to cluster'
      };
    }

    this.logger.membership('Manually adding node to cluster', {
      nodeAddress: nodeAddress.toString()
    });

    try {
      const response = await this.membershipManager.handleMembershipApplication(
        nodeAddress.toObject(),
        this.membershipManager.getClusterInfo().term
      );

      return {
        success: response.success,
        message: response.message || (response.success ? 'Node added successfully' : 'Failed to add node')
      };

    } catch (error) {
      this.logger.error('Failed to manually add node', error as Error, {
        nodeAddress: nodeAddress.toString()
      });

      return {
        success: false,
        message: (error as Error).message
      };
    }
  }

  /**
   * Manually remove node from cluster (admin operation)
   */
  async removeNode(nodeAddress: Address): Promise<{ success: boolean; message: string }> {
    if (!this.raftNode.isLeader()) {
      return {
        success: false,
        message: 'Only leader can remove nodes from cluster'
      };
    }

    this.logger.membership('Manually removing node from cluster', {
      nodeAddress: nodeAddress.toString()
    });

    try {
      const success = await this.membershipManager.removeNodeFromCluster(nodeAddress);

      return {
        success,
        message: success ? 'Node removed successfully' : 'Failed to remove node'
      };

    } catch (error) {
      this.logger.error('Failed to manually remove node', error as Error, {
        nodeAddress: nodeAddress.toString()
      });

      return {
        success: false,
        message: (error as Error).message
      };
    }
  }

  /**
   * Update member last seen (called from heartbeat responses)
   */
  updateMemberLastSeen(memberAddress: Address): void {
    this.membershipManager.updateMemberLastSeen(memberAddress);
  }

  /**
   * Get membership statistics
   */
  getMembershipStats(): {
    clusterSize: number;
    activeMembers: number;
    suspectedMembers: number;
    failedMembers: number;
    leader: string | null;
    self: string;
    memberDetails: Array<{
      address: string;
      nodeId: string;
      status: string;
      role: string;
      lastSeen: number;
    }>;
  } {
    const clusterInfo = this.membershipManager.getClusterInfo();

    const activeMembers = clusterInfo.memberDetails.filter(m => m.status === 'active').length;
    const suspectedMembers = clusterInfo.memberDetails.filter(m => m.status === 'suspected').length;
    const failedMembers = clusterInfo.memberDetails.filter(m => m.status === 'failed').length;

    return {
      clusterSize: clusterInfo.size,
      activeMembers,
      suspectedMembers,
      failedMembers,
      leader: clusterInfo.leader ? `${clusterInfo.leader.ip}:${clusterInfo.leader.port}` : null,
      self: `${clusterInfo.self.ip}:${clusterInfo.self.port}`,
      memberDetails: clusterInfo.memberDetails
    };
  }

  /**
   * Check if node is healthy and part of cluster
   */
  isHealthy(): boolean {
    const clusterInfo = this.membershipManager.getClusterInfo();
    return clusterInfo.size > 0 && this.raftNode.isLeader() || this.membershipManager.getLeaderAddress() !== null;
  }

  /**
   * Get integration status
   */
  getIntegrationStatus(): {
    isIntegrated: boolean;
    raftNodeRole: string;
    membershipManagerLeader: string | null;
    clusterSize: number;
    lastUpdate: number;
  } {
    const raftClusterInfo = this.raftNode.getClusterInfo();
    const membershipClusterInfo = this.membershipManager.getClusterInfo();

    return {
      isIntegrated: true,
      raftNodeRole: raftClusterInfo.type,
      membershipManagerLeader: membershipClusterInfo.leader?.toString() || null,
      clusterSize: membershipClusterInfo.size,
      lastUpdate: Date.now()
    };
  }

  /**
   * Cleanup integration
   */
  cleanup(): void {
    this.logger.membership('Cleaning up membership integration');
    
    // Remove event listeners
    this.raftNode.removeAllListeners();
    this.membershipManager.removeAllListeners();
    
    this.membershipManager.cleanup();
  }
}