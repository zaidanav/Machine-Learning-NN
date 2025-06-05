/**
 * FIXED RaftNode.ts - Complete Implementation with Working Membership Removal
 * All bugs fixed, especially the Address comparison issue in removeNodeFromCluster
 */

import { Address, AddressData } from './Address';
import { Logger } from '../utils/Logger';
import { EventEmitter } from 'events';
import { KVStore, KVCommand, KVResult } from '../app/KVStore';

// Raft Node Types
export enum NodeType {
  LEADER = 'LEADER',
  FOLLOWER = 'FOLLOWER',
  CANDIDATE = 'CANDIDATE'
}

// Raft RPC Messages
export interface VoteRequest {
  term: number;
  candidateId: AddressData;
  lastLogIndex: number;
  lastLogTerm: number;
}

export interface VoteResponse {
  term: number;
  voteGranted: boolean;
  voterId: AddressData;
}

export interface HeartbeatRequest {
  term: number;
  leaderId: AddressData;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
  clusterMembers?: AddressData[];
}

export interface HeartbeatResponse {
  term: number;
  success: boolean;
  nodeId: AddressData;
  matchIndex?: number;
}

export interface LogEntry {
  term: number;
  index: number;
  command: string;
  data: any;
  timestamp: number;
  id: string;
}

export interface MembershipRequest {
  nodeAddr: AddressData;
  term: number;
}

export interface MembershipResponse {
  status: 'success' | 'redirected' | 'error';
  leaderAddr?: AddressData;
  clusterAddrList?: AddressData[];
  term?: number;
  message?: string;
}

export interface LogReplicationRequest {
  term: number;
  leaderId: AddressData;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
}

export interface LogReplicationResponse {
  term: number;
  success: boolean;
  nodeId: AddressData;
  matchIndex: number;
  conflictIndex?: number;
  conflictTerm?: number;
}

export interface ClusterInfo {
  address: AddressData;
  type: NodeType;
  currentTerm: number;
  clusterSize: number;
  leader?: AddressData;
  votedFor?: AddressData;
  lastLogIndex: number;
  commitIndex: number;
}

export interface MembershipUpdateRequest {
  type: 'add' | 'remove' | 'sync';
  nodeAddr: AddressData;
  clusterState: AddressData[];
  term: number;
  timestamp: number;
}

export interface MembershipUpdateResponse {
  success: boolean;
  message?: string;
  newClusterSize?: number;
}

/**
 * Core Raft Node Implementation with FIXED Membership Management
 */
export class RaftNode extends EventEmitter {
  // Raft timing configuration
  private static readonly HEARTBEAT_INTERVAL = 1000;
  private static readonly ELECTION_TIMEOUT_MIN = 3000;
  private static readonly ELECTION_TIMEOUT_MAX = 6000;
  private static readonly RPC_TIMEOUT = 1000;
  private static readonly LOG_REPLICATION_TIMEOUT = 2000;

  // Node state
  private address: Address;
  private nodeType: NodeType = NodeType.FOLLOWER;
  private currentTerm: number = 0;
  private votedFor: Address | null = null;
  private log: LogEntry[] = [];
  private commitIndex: number = -1;
  private lastApplied: number = -1;

  // FIXED: Enhanced cluster state management with proper tracking
  private clusterAddresses: Set<Address> = new Set();
  private clusterMembers: Map<string, {
    address: Address;
    lastSeen: number;
    status: 'active' | 'suspected' | 'failed' | 'removing';
    consecutiveFailures: number;
  }> = new Map();
  private leaderAddress: Address | null = null;

  // Leader state
  private nextIndex: Map<string, number> = new Map();
  private matchIndex: Map<string, number> = new Map();

  // Timers and control
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private electionTimer: NodeJS.Timeout | null = null;
  private running: boolean = false;

  // Election state
  private votesReceived: Set<string> = new Set();
  private lastHeartbeatTime: number = Date.now();

  // Log Replication state
  private pendingEntries: Map<string, { resolve: Function; reject: Function; timestamp: number }> = new Map();
  private replicationInProgress: Set<string> = new Set();

  // Dependencies
  private logger: Logger;
  private rpcClient: any;
  private kvStore: KVStore;

  constructor(
    address: Address,
    kvStore: KVStore,
    logger?: Logger,
    rpcClient?: any
  ) {
    super();
    
    this.address = address;
    this.kvStore = kvStore;
    this.logger = logger || new Logger({ 
      address, 
      nodeId: process.env.NODE_ID 
    });
    this.rpcClient = rpcClient;

    // FIXED: Initialize self in cluster with proper tracking
    this.addMemberToCluster(address, true);
    
    this.logger.info('Raft node initialized with fixed membership', { 
      address: address.toString(),
      nodeType: this.nodeType 
    });
  }

  /**
   * FIXED: Enhanced member management
   */
  private addMemberToCluster(address: Address, isSelf: boolean = false): void {
    this.clusterAddresses.add(address);
    this.clusterMembers.set(address.hashCode(), {
      address,
      lastSeen: Date.now(),
      status: 'active',
      consecutiveFailures: 0
    });

    this.logger.membership('Member added to cluster', {
      address: address.toString(),
      isSelf,
      clusterSize: this.clusterAddresses.size
    });
  }

  /**
   * FIXED: removeNodeFromCluster - The main fix for membership removal
   */
  async removeNodeFromCluster(nodeAddress: Address): Promise<boolean> {
    this.logger.info('🔧 FIXED removeNodeFromCluster called', {
      nodeAddress: nodeAddress.toString(),
      nodeType: this.nodeType,
      clusterSize: this.clusterAddresses.size
    });

    // FIXED: Only leader can remove nodes
    if (this.nodeType !== NodeType.LEADER) {
      this.logger.warn('Cannot remove node: not the leader', {
        nodeAddress: nodeAddress.toString(),
        currentRole: this.nodeType
      });
      return false;
    }

    // FIXED: Cannot remove self
    if (nodeAddress.equals(this.address)) {
      this.logger.warn('Cannot remove self from cluster', {
        nodeAddress: nodeAddress.toString(),
        selfAddress: this.address.toString()
      });
      return false;
    }

    // FIXED: Log current cluster state for debugging
    this.logger.info('🔍 Current cluster before removal', {
      targetNode: nodeAddress.toString(),
      clusterMembers: Array.from(this.clusterAddresses).map(a => a.toString()),
      clusterSize: this.clusterAddresses.size
    });

    // FIXED: Proper node existence check with detailed logging
    let targetAddressToRemove: Address | null = null;
    for (const addr of this.clusterAddresses) {
      this.logger.debug('🔍 Comparing addresses', {
        clusterAddr: addr.toString(),
        targetAddr: nodeAddress.toString(),
        clusterAddrType: typeof addr,
        targetAddrType: typeof nodeAddress,
        equals: addr.equals(nodeAddress)
      });
      
      if (addr.equals(nodeAddress)) {
        targetAddressToRemove = addr;
        break;
      }
    }

    if (!targetAddressToRemove) {
      this.logger.warn('❌ Node not found in cluster', {
        nodeAddress: nodeAddress.toString(),
        currentMembers: Array.from(this.clusterAddresses).map(a => a.toString()),
        searchedFor: `${nodeAddress.ip}:${nodeAddress.port}`
      });
      return false;
    }

    try {
      // FIXED: Notify the node about removal (best effort)
      try {
        await this.notifyNodeRemoval(targetAddressToRemove, 'manual_removal');
      } catch (error) {
        this.logger.warn('Could not notify node about removal', {
          nodeAddress: targetAddressToRemove.toString(),
          error: (error as Error).message
        });
        // Continue with removal even if notification fails
      }

      // FIXED: Remove from cluster addresses using the exact object reference
      const removed = this.clusterAddresses.delete(targetAddressToRemove);
      
      // FIXED: Remove from member tracking
      const memberKey = targetAddressToRemove.hashCode();
      this.clusterMembers.delete(memberKey);
      
      // FIXED: Clean up leader state
      this.nextIndex.delete(memberKey);
      this.matchIndex.delete(memberKey);

      this.logger.info('✅ Remove operation result', {
        nodeAddress: targetAddressToRemove.toString(),
        removed,
        newClusterSize: this.clusterAddresses.size,
        newMembers: Array.from(this.clusterAddresses).map(a => a.toString())
      });

      if (removed) {
        // FIXED: Broadcast removal to all remaining nodes
        await this.broadcastMembershipChange('remove', targetAddressToRemove);

        // FIXED: Create a log entry for the membership change
        const membershipLogEntry: LogEntry = {
          term: this.currentTerm,
          index: this.log.length,
          command: 'membership_remove',
          data: {
            nodeAddress: targetAddressToRemove.toObject(),
            timestamp: Date.now(),
            initiator: this.address.toObject()
          },
          timestamp: Date.now(),
          id: this.generateEntryId()
        };

        this.log.push(membershipLogEntry);
        
        // Try to replicate membership change
        try {
          await this.replicateLogEntry(membershipLogEntry);
          this.commitIndex = membershipLogEntry.index;
        } catch (error) {
          this.logger.warn('Failed to replicate membership change', {
            error: (error as Error).message
          });
        }

        this.logger.membership('✅ Node successfully removed from cluster', {
          nodeAddress: targetAddressToRemove.toString(),
          newClusterSize: this.clusterAddresses.size,
          remainingNodes: Array.from(this.clusterAddresses).map(a => a.toString())
        });

        this.emit('memberRemoved', { 
          address: targetAddressToRemove, 
          clusterSize: this.clusterAddresses.size 
        });

        return true;
      } else {
        this.logger.error('❌ Set.delete() returned false - this should not happen');
        return false;
      }

    } catch (error) {
      this.logger.error('❌ Failed to remove node from cluster', error as Error, {
        nodeAddress: nodeAddress.toString()
      });
      return false;
    }
  }

  /**
   * FIXED: Enhanced notification system for node removal
   */
  private async notifyNodeRemoval(nodeAddress: Address, reason: string): Promise<void> {
    try {
      await this.sendRPC(nodeAddress, 'membershipRemovalNotification', {
        reason,
        term: this.currentTerm,
        removedBy: this.address.toObject(),
        timestamp: Date.now()
      });

      this.logger.membership('Removal notification sent', {
        nodeAddress: nodeAddress.toString(),
        reason
      });

    } catch (error) {
      this.logger.warn('Failed to send removal notification', {
        nodeAddress: nodeAddress.toString(),
        error: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * FIXED: Enhanced membership change broadcasting
   */
  private async broadcastMembershipChange(
    changeType: 'add' | 'remove',
    targetAddress: Address
  ): Promise<void> {
    const membershipUpdate: MembershipUpdateRequest = {
      type: changeType,
      nodeAddr: targetAddress.toObject(),
      clusterState: Array.from(this.clusterAddresses).map(addr => addr.toObject()),
      term: this.currentTerm,
      timestamp: Date.now()
    };

    const broadcastPromises: Promise<void>[] = [];

    for (const memberAddress of this.clusterAddresses) {
      // Skip self and target node
      if (memberAddress.equals(this.address) || memberAddress.equals(targetAddress)) {
        continue;
      }

      broadcastPromises.push(
        this.sendMembershipUpdate(memberAddress, membershipUpdate)
      );
    }

    if (broadcastPromises.length > 0) {
      const results = await Promise.allSettled(broadcastPromises);
      
      const successCount = results.filter(r => r.status === 'fulfilled').length;
      const failureCount = results.filter(r => r.status === 'rejected').length;

      this.logger.membership('Membership change broadcast completed', {
        changeType,
        targetAddress: targetAddress.toString(),
        successCount,
        failureCount,
        totalNotifications: broadcastPromises.length
      });
    }
  }

  /**
   * FIXED: Enhanced membership update sending
   */
  private async sendMembershipUpdate(
    nodeAddress: Address,
    updateRequest: MembershipUpdateRequest
  ): Promise<void> {
    try {
      const response = await this.sendRPC<MembershipUpdateResponse>(
        nodeAddress,
        'membershipUpdate',
        updateRequest
      );

      if (!response.success) {
        throw new Error(response.message || 'Membership update failed');
      }

      this.logger.debug('Membership update sent successfully', {
        nodeAddress: nodeAddress.toString(),
        updateType: updateRequest.type
      });

    } catch (error) {
      this.logger.warn('Failed to send membership update', {
        nodeAddress: nodeAddress.toString(),
        error: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * Start the Raft node
   */
  async start(contactAddress?: Address): Promise<void> {
    if (this.running) {
      throw new Error('Node is already running');
    }

    this.running = true;
    this.logger.info('Starting Raft node...');

    try {
      if (contactAddress) {
        // Join existing cluster
        await this.joinCluster(contactAddress);
        this.startElectionTimeout();
      } else {
        // Bootstrap new cluster as leader
        await this.becomeLeader();
      }

      this.logger.info('Raft node started successfully', {
        nodeType: this.nodeType,
        term: this.currentTerm,
        clusterSize: this.clusterAddresses.size
      });

    } catch (error) {
      this.running = false;
      this.logger.error('Failed to start Raft node', error as Error);
      throw error;
    }
  }

  /**
   * Stop the Raft node gracefully
   */
  async stop(): Promise<void> {
    if (!this.running) {
      return;
    }

    this.logger.info('Stopping Raft node...');
    this.running = false;

    // Clear timers
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.electionTimer) {
      clearTimeout(this.electionTimer);
      this.electionTimer = null;
    }

    // Reject pending entries
    this.pendingEntries.forEach(({ reject }) => {
      reject(new Error('Node is shutting down'));
    });
    this.pendingEntries.clear();

    this.logger.info('Raft node stopped');
  }

  /**
   * Execute client command with log replication
   */
  async execute(command: string): Promise<any> {
    if (this.nodeType !== NodeType.LEADER) {
      throw new Error('Not the leader - redirect to: ' + this.leaderAddress?.toString());
    }

    try {
      // Parse command
      const kvCommand: KVCommand = KVStore.parseCommand(command);
      
      // Handle ping immediately (no log replication needed)
      if (kvCommand.type === 'ping') {
        return await this.kvStore.executeCommand(kvCommand);
      }

      // Create log entry for state-changing commands
      if (['set', 'del', 'append'].includes(kvCommand.type)) {
        return await this.executeWithLogReplication(kvCommand);
      } else {
        // Read-only commands (get, strln) - execute directly for consistency
        return await this.kvStore.executeCommand(kvCommand);
      }

    } catch (error) {
      this.logger.error('Command execution failed', error as Error, { command });
      throw error;
    }
  }

  /**
   * Execute command with log replication
   */
  private async executeWithLogReplication(kvCommand: KVCommand): Promise<KVResult> {
    const entryId = this.generateEntryId();
    
    // Create log entry
    const logEntry: LogEntry = {
      term: this.currentTerm,
      index: this.log.length,
      command: kvCommand.type,
      data: {
        key: kvCommand.key,
        value: kvCommand.value,
        originalCommand: kvCommand
      },
      timestamp: Date.now(),
      id: entryId
    };

    this.logger.logReplication('Creating log entry', {
      entryId,
      command: kvCommand.type,
      key: kvCommand.key,
      index: logEntry.index
    });

    // Add to log
    this.log.push(logEntry);

    // Replicate to majority of followers
    try {
      const success = await this.replicateLogEntry(logEntry);
      
      if (success) {
        // Commit and apply to state machine
        this.commitIndex = logEntry.index;
        await this.applyLogEntries();
        
        this.logger.logReplication('Command executed successfully', {
          entryId,
          index: logEntry.index,
          commitIndex: this.commitIndex
        });

        return {
          success: true,
          result: 'OK'
        };
      } else {
        // Remove from log if replication failed
        this.log.pop();
        throw new Error('Failed to replicate to majority of nodes');
      }

    } catch (error) {
      // Remove from log if replication failed
      if (this.log[this.log.length - 1]?.id === entryId) {
        this.log.pop();
      }
      
      this.logger.error('Log replication failed', error as Error, { entryId });
      throw error;
    }
  }

  /**
   * Replicate log entry to followers
   */
  private async replicateLogEntry(entry: LogEntry): Promise<boolean> {
    const followers = Array.from(this.clusterAddresses).filter(addr => !addr.equals(this.address));
    
    if (followers.length === 0) {
      // Single node cluster - always succeed
      this.logger.logReplication('Single node cluster, auto-commit', {
        entryIndex: entry.index
      });
      return true;
    }

    this.logger.logReplication('Replicating log entry to followers', {
      entryIndex: entry.index,
      followersCount: followers.length,
      clusterSize: this.clusterAddresses.size
    });

    const replicationPromises = followers.map(follower => 
      this.replicateToFollowerWithRetry(follower, [entry])
    );

    try {
      const results = await Promise.allSettled(replicationPromises);
      const successCount = results.filter(result => 
        result.status === 'fulfilled' && result.value === true
      ).length;

      // Calculate majority correctly
      const totalNodes = this.clusterAddresses.size;
      const requiredSuccess = Math.floor(totalNodes / 2);
      const totalSuccess = successCount + 1; // +1 for leader

      this.logger.logReplication('Replication results', {
        entryIndex: entry.index,
        successCount,
        totalSuccess,
        requiredSuccess,
        totalNodes,
        passed: totalSuccess > requiredSuccess
      });

      return totalSuccess > requiredSuccess;

    } catch (error) {
      this.logger.error('Error during log replication', error as Error);
      return false;
    }
  }

  /**
   * Join existing cluster
   */
  private async joinCluster(contactAddress: Address): Promise<void> {
    this.logger.info('Attempting to join cluster', { 
      contactAddress: contactAddress.toString() 
    });

    const request: MembershipRequest = {
      nodeAddr: this.address.toObject(),
      term: this.currentTerm
    };

    try {
      const response = await this.sendRPC<MembershipResponse>(
        contactAddress, 
        'applyMembership', 
        request
      );

      if (response.status === 'success' && response.clusterAddrList && response.leaderAddr) {
        // Clear and rebuild cluster information from leader
        this.clusterAddresses.clear();
        this.clusterMembers.clear();
        
        response.clusterAddrList.forEach(addr => {
          const address = Address.fromObject(addr);
          this.addMemberToCluster(address, address.equals(this.address));
        });

        this.leaderAddress = Address.fromObject(response.leaderAddr);
        this.currentTerm = response.term || 0;

        this.logger.info('Successfully joined cluster', {
          leader: this.leaderAddress.toString(),
          clusterSize: this.clusterAddresses.size,
          term: this.currentTerm
        });

      } else if (response.status === 'redirected' && response.leaderAddr) {
        const actualLeader = Address.fromObject(response.leaderAddr);
        this.logger.info('Redirected to actual leader', { 
          leader: actualLeader.toString() 
        });
        await this.joinCluster(actualLeader);
      } else {
        throw new Error(`Failed to join cluster: ${response.message || 'Unknown error'}`);
      }

    } catch (error) {
      this.logger.error('Error joining cluster', error as Error);
      throw error;
    }
  }

  /**
   * Become leader (for bootstrap or after election)
   */
  private async becomeLeader(): Promise<void> {
    this.logger.election('Becoming leader', { term: this.currentTerm + 1 });
    
    this.nodeType = NodeType.LEADER;
    this.currentTerm++;
    this.leaderAddress = this.address;
    this.votedFor = null;

    // Initialize leader state
    this.nextIndex.clear();
    this.matchIndex.clear();
    
    for (const addr of this.clusterAddresses) {
      if (!addr.equals(this.address)) {
        this.nextIndex.set(addr.hashCode(), this.log.length);
        this.matchIndex.set(addr.hashCode(), -1);
      }
    }

    // Start sending heartbeats
    this.startHeartbeat();

    this.logger.updateContext({ 
      nodeType: NodeType.LEADER,
      term: this.currentTerm 
    });

    this.emit('leadershipAcquired', {
      term: this.currentTerm,
      address: this.address
    });

    this.logger.election('Successfully became leader', {
      term: this.currentTerm,
      clusterSize: this.clusterAddresses.size
    });
  }

  /**
   * Become follower
   */
  private becomeFollower(term: number, leader?: Address): void {
    const wasLeader = this.nodeType === NodeType.LEADER;
    
    this.logger.election('Becoming follower', { 
      currentTerm: this.currentTerm,
      newTerm: term,
      leader: leader?.toString()
    });

    this.nodeType = NodeType.FOLLOWER;
    this.currentTerm = term;
    this.votedFor = null;
    this.lastHeartbeatTime = Date.now();

    if (leader) {
      this.leaderAddress = leader;
    }

    if (wasLeader && this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    this.startElectionTimeout();

    this.logger.updateContext({ 
      nodeType: NodeType.FOLLOWER,
      term: this.currentTerm 
    });

    this.emit('becameFollower', {
      term: this.currentTerm,
      leader: leader
    });
  }

  /**
   * Enhanced membership application handling
   */
  async handleMembershipApplication(request: MembershipRequest): Promise<MembershipResponse> {
    const newNodeAddr = Address.fromObject(request.nodeAddr);
    
    this.logger.membership('Received membership application', {
      from: newNodeAddr.toString(),
      currentClusterSize: this.clusterAddresses.size,
      isLeader: this.nodeType === NodeType.LEADER
    });

    if (this.nodeType !== NodeType.LEADER) {
      return {
        status: 'redirected',
        leaderAddr: this.leaderAddress?.toObject(),
        term: this.currentTerm,
        message: 'Not the leader'
      };
    }

    try {
      // Add node if not already in cluster
      let alreadyExists = false;
      for (const addr of this.clusterAddresses) {
        if (addr.equals(newNodeAddr)) {
          alreadyExists = true;
          break;
        }
      }

      if (!alreadyExists) {
        this.addMemberToCluster(newNodeAddr);
        
        if (this.nodeType === NodeType.LEADER) {
          this.nextIndex.set(newNodeAddr.hashCode(), this.log.length);
          this.matchIndex.set(newNodeAddr.hashCode(), -1);
        }

        // Broadcast addition to other nodes
        await this.broadcastMembershipChange('add', newNodeAddr);
      }

      // Return current cluster state
      return {
        status: 'success',
        clusterAddrList: Array.from(this.clusterAddresses).map(addr => addr.toObject()),
        leaderAddr: this.address.toObject(),
        term: this.currentTerm,
        message: 'Successfully joined cluster'
      };

    } catch (error) {
      this.logger.error('Failed to process membership application', error as Error);
      return {
        status: 'error',
        term: this.currentTerm,
        message: (error as Error).message
      };
    }
  }

  /**
   * Handle membership update notification
   */
  async handleMembershipUpdate(updateRequest: MembershipUpdateRequest): Promise<MembershipUpdateResponse> {
    this.logger.membership('Received membership update', {
      type: updateRequest.type,
      nodeAddr: `${updateRequest.nodeAddr.ip}:${updateRequest.nodeAddr.port}`,
      term: updateRequest.term,
      clusterStateSize: updateRequest.clusterState.length
    });

    try {
      // Update term if necessary
      if (updateRequest.term > this.currentTerm) {
        this.becomeFollower(updateRequest.term);
      }

      const targetAddress = Address.fromObject(updateRequest.nodeAddr);

      if (updateRequest.type === 'add') {
        let alreadyExists = false;
        for (const addr of this.clusterAddresses) {
          if (addr.equals(targetAddress)) {
            alreadyExists = true;
            break;
          }
        }
        if (!alreadyExists) {
          this.addMemberToCluster(targetAddress);
        }
      } else if (updateRequest.type === 'remove') {
        // Find and remove the exact address object
        let addressToRemove: Address | null = null;
        for (const addr of this.clusterAddresses) {
          if (addr.equals(targetAddress)) {
            addressToRemove = addr;
            break;
          }
        }
        if (addressToRemove) {
          this.clusterAddresses.delete(addressToRemove);
          this.clusterMembers.delete(addressToRemove.hashCode());
        }
      } else if (updateRequest.type === 'sync') {
        // Full cluster state sync
        this.syncClusterState(updateRequest.clusterState, updateRequest.term);
      }

      return {
        success: true,
        message: `Membership update applied: ${updateRequest.type}`,
        newClusterSize: this.clusterAddresses.size
      };

    } catch (error) {
      this.logger.error('Failed to handle membership update', error as Error);
      return {
        success: false,
        message: (error as Error).message
      };
    }
  }

  /**
   * Cluster state synchronization
   */
  private syncClusterState(clusterState: AddressData[], term: number): void {
    this.logger.membership('Syncing cluster state', {
      currentSize: this.clusterAddresses.size,
      newSize: clusterState.length,
      term
    });

    const selfAddress = this.address;
    
    // Clear and rebuild cluster state
    this.clusterAddresses.clear();
    this.clusterMembers.clear();
    this.nextIndex.clear();
    this.matchIndex.clear();

    // Add all nodes from new state
    clusterState.forEach(addrData => {
      const address = Address.fromObject(addrData);
      this.addMemberToCluster(address, address.equals(selfAddress));
      
      if (this.nodeType === NodeType.LEADER && !address.equals(selfAddress)) {
        this.nextIndex.set(address.hashCode(), this.log.length);
        this.matchIndex.set(address.hashCode(), -1);
      }
    });

    // Update term
    if (term > this.currentTerm) {
      this.currentTerm = term;
    }

    this.logger.membership('Cluster state synchronized', {
      newSize: this.clusterAddresses.size,
      term: this.currentTerm
    });
  }

  /**
   * Get current cluster information
   */
  getClusterInfo(): ClusterInfo & {
    memberDetails?: Array<{
      address: string;
      status: string;
      lastSeen: number;
      consecutiveFailures: number;
    }>;
  } {
    const baseInfo: ClusterInfo = {
      address: this.address.toObject(),
      type: this.nodeType,
      currentTerm: this.currentTerm,
      clusterSize: this.clusterAddresses.size,
      leader: this.leaderAddress?.toObject(),
      votedFor: this.votedFor?.toObject(),
      lastLogIndex: this.log.length - 1,
      commitIndex: this.commitIndex
    };

    // Add membership details
    const memberDetails = Array.from(this.clusterMembers.values()).map(member => ({
      address: member.address.toString(),
      status: member.status,
      lastSeen: member.lastSeen,
      consecutiveFailures: member.consecutiveFailures
    }));

    return {
      ...baseInfo,
      memberDetails
    };
  }

  /**
   * Check if this node is the leader
   */
  isLeader(): boolean {
    return this.nodeType === NodeType.LEADER;
  }

  /**
   * Get current leader address
   */
  getLeaderAddress(): Address | null {
    return this.leaderAddress;
  }

  /**
   * Get cluster addresses
   */
  getClusterAddresses(): Address[] {
    return Array.from(this.clusterAddresses);
  }

  /**
   * Check if node is in cluster
   */
  isMemberOfCluster(nodeAddress: Address): boolean {
    for (const addr of this.clusterAddresses) {
      if (addr.equals(nodeAddress)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get membership statistics
   */
  getMembershipStats(): {
    totalMembers: number;
    activeMembers: number;
    suspectedMembers: number;
    failedMembers: number;
    removingMembers: number;
  } {
    const stats = {
      totalMembers: this.clusterMembers.size,
      activeMembers: 0,
      suspectedMembers: 0,
      failedMembers: 0,
      removingMembers: 0
    };

    for (const member of this.clusterMembers.values()) {
      switch (member.status) {
        case 'active':
          stats.activeMembers++;
          break;
        case 'suspected':
          stats.suspectedMembers++;
          break;
        case 'failed':
          stats.failedMembers++;
          break;
        case 'removing':
          stats.removingMembers++;
          break;
      }
    }

    return stats;
  }

  /**
   * Get current log (for debugging and demo)
   */
  getLog(): LogEntry[] {
    return [...this.log];
  }

  /**
   * Get detailed log information for request_log handler
   */
  getDetailedLog(): any {
    return {
      entries: this.log.map(entry => ({
        term: entry.term,
        index: entry.index,
        command: entry.command,
        key: entry.data?.key,
        value: entry.data?.value,
        timestamp: new Date(entry.timestamp).toISOString(),
        id: entry.id
      })),
      commitIndex: this.commitIndex,
      lastApplied: this.lastApplied,
      logLength: this.log.length,
      clusterInfo: this.getClusterInfo()
    };
  }

  /**
   * Handle vote request from candidate
   */
  async handleVoteRequest(request: VoteRequest): Promise<VoteResponse> {
    const candidateAddr = Address.fromObject(request.candidateId);
    
    this.logger.election('Received vote request', {
      from: candidateAddr.toString(),
      candidateTerm: request.term,
      currentTerm: this.currentTerm
    });

    let voteGranted = false;

    if (request.term > this.currentTerm) {
      this.becomeFollower(request.term);
    }

    if (request.term === this.currentTerm && 
        (this.votedFor === null || this.votedFor.equals(candidateAddr))) {
      
      const lastLogTerm = this.log.length > 0 ? this.log[this.log.length - 1]!.term : 0;
      const lastLogIndex = this.log.length - 1;

      if (request.lastLogTerm > lastLogTerm || 
          (request.lastLogTerm === lastLogTerm && request.lastLogIndex >= lastLogIndex)) {
        
        voteGranted = true;
        this.votedFor = candidateAddr;
        this.logger.election('Vote granted', { to: candidateAddr.toString() });
      }
    }

    if (!voteGranted) {
      this.logger.election('Vote denied', { 
        to: candidateAddr.toString(),
        reason: 'conditions not met'
      });
    }

    return {
      term: this.currentTerm,
      voteGranted,
      voterId: this.address.toObject()
    };
  }

  /**
   * Handle heartbeat from leader
   */
  async handleHeartbeat(request: HeartbeatRequest): Promise<HeartbeatResponse> {
    const leaderAddr = Address.fromObject(request.leaderId);
    
    this.logger.heartbeat('Received heartbeat', {
      from: leaderAddr.toString(),
      leaderTerm: request.term,
      currentTerm: this.currentTerm,
      hasClusterMembers: !!request.clusterMembers,
      entriesCount: request.entries.length
    });

    let success = false;
    let matchIndex = this.log.length - 1;

    if (request.term >= this.currentTerm) {
      if (this.nodeType !== NodeType.FOLLOWER) {
        this.becomeFollower(request.term, leaderAddr);
      } else {
        this.currentTerm = request.term;
        this.leaderAddress = leaderAddr;
      }
      
      this.lastHeartbeatTime = Date.now();
      
      // Sync cluster membership if provided by leader
      if (request.clusterMembers && request.clusterMembers.length > 0) {
        const newClusterSize = request.clusterMembers.length;
        const currentClusterSize = this.clusterAddresses.size;
        
        if (newClusterSize !== currentClusterSize) {
          this.logger.membership('Syncing cluster membership from leader', {
            currentSize: currentClusterSize,
            newSize: newClusterSize
          });
          
          this.syncClusterState(request.clusterMembers, request.term);
        }
      }

      // Handle log replication if entries are included
      if (request.entries.length > 0) {
        const replicationResponse = await this.handleLogReplication({
          term: request.term,
          leaderId: request.leaderId,
          prevLogIndex: request.prevLogIndex,
          prevLogTerm: request.prevLogTerm,
          entries: request.entries,
          leaderCommit: request.leaderCommit
        });
        
        success = replicationResponse.success;
        matchIndex = replicationResponse.matchIndex;
      } else {
        success = true;
      }
    }

    return {
      term: this.currentTerm,
      success,
      nodeId: this.address.toObject(),
      matchIndex
    };
  }

  /**
   * Handle log replication request from leader
   */
  async handleLogReplication(request: LogReplicationRequest): Promise<LogReplicationResponse> {
    const leaderAddr = Address.fromObject(request.leaderId);
    
    this.logger.logReplication('Received log replication request', {
      from: leaderAddr.toString(),
      term: request.term,
      entries: request.entries.length,
      prevLogIndex: request.prevLogIndex
    });

    // Check term
    if (request.term < this.currentTerm) {
      return {
        term: this.currentTerm,
        success: false,
        nodeId: this.address.toObject(),
        matchIndex: this.log.length - 1
      };
    }

    // Update term and leader if necessary
    if (request.term > this.currentTerm) {
      this.becomeFollower(request.term, leaderAddr);
    } else if (this.nodeType !== NodeType.FOLLOWER) {
      this.becomeFollower(request.term, leaderAddr);
    }

    this.lastHeartbeatTime = Date.now();

    // Check log consistency
    if (request.prevLogIndex >= 0) {
      if (request.prevLogIndex >= this.log.length ||
          this.log[request.prevLogIndex]?.term !== request.prevLogTerm) {
        
        this.logger.logReplication('Log inconsistency detected', {
          prevLogIndex: request.prevLogIndex,
          prevLogTerm: request.prevLogTerm,
          actualTerm: this.log[request.prevLogIndex]?.term,
          logLength: this.log.length
        });

        return {
          term: this.currentTerm,
          success: false,
          nodeId: this.address.toObject(),
          matchIndex: Math.min(request.prevLogIndex - 1, this.log.length - 1),
          conflictIndex: request.prevLogIndex,
          conflictTerm: this.log[request.prevLogIndex]?.term
        };
      }
    }

    // Append new entries
    if (request.entries.length > 0) {
      // Remove conflicting entries
      this.log = this.log.slice(0, request.prevLogIndex + 1);
      
      // Append new entries
      this.log.push(...request.entries);
      
      this.logger.logReplication('Appended log entries', {
        entriesCount: request.entries.length,
        newLogLength: this.log.length
      });
    }

    // Update commit index
    if (request.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(request.leaderCommit, this.log.length - 1);
      await this.applyLogEntries();
    }

    return {
      term: this.currentTerm,
      success: true,
      nodeId: this.address.toObject(),
      matchIndex: this.log.length - 1
    };
  }

  /**
   * Apply committed log entries to state machine
   */
  private async applyLogEntries(): Promise<void> {
    while (this.lastApplied < this.commitIndex) {
      this.lastApplied++;
      const entry = this.log[this.lastApplied];
      
      if (entry && entry.data.originalCommand) {
        try {
          await this.kvStore.executeCommand(entry.data.originalCommand);
          
          this.logger.logReplication('Applied log entry to state machine', {
            index: entry.index,
            command: entry.command,
            key: entry.data.key
          });
          
        } catch (error) {
          this.logger.error('Failed to apply log entry', error as Error, {
            index: entry.index,
            command: entry.command
          });
        }
      }
    }
  }

  /**
   * Start heartbeat timer (leader only)
   */
  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.nodeType === NodeType.LEADER && this.running) {
        this.sendHeartbeats();
      }
    }, RaftNode.HEARTBEAT_INTERVAL);
  }

  /**
   * Send heartbeats to all followers
   */
  private async sendHeartbeats(): Promise<void> {
    if (this.nodeType !== NodeType.LEADER) {
      return;
    }

    this.logger.heartbeat('Sending heartbeats to followers', {
      followers: this.clusterAddresses.size - 1,
      term: this.currentTerm,
      clusterSize: this.clusterAddresses.size
    });

    const heartbeatPromises: Promise<void>[] = [];

    for (const addr of this.clusterAddresses) {
      if (!addr.equals(this.address)) {
        heartbeatPromises.push(this.sendHeartbeatToNode(addr));
      }
    }

    await Promise.allSettled(heartbeatPromises);
  }

  /**
   * Send heartbeat to specific node
   */
  private async sendHeartbeatToNode(address: Address): Promise<void> {
    try {
      const prevLogIndex = this.nextIndex.get(address.hashCode()) || 0;
      const prevLogTerm = prevLogIndex > 0 ? this.log[prevLogIndex - 1]?.term || 0 : 0;

      const heartbeat: HeartbeatRequest = {
        term: this.currentTerm,
        leaderId: this.address.toObject(),
        prevLogIndex: prevLogIndex - 1,
        prevLogTerm,
        entries: [],
        leaderCommit: this.commitIndex,
        clusterMembers: Array.from(this.clusterAddresses).map(addr => addr.toObject())
      };

      const response = await this.sendRPC<HeartbeatResponse>(address, 'heartbeat', heartbeat);
      this.handleHeartbeatResponse(response, address);

    } catch (error) {
      this.logger.warn('Failed to send heartbeat', {
        target: address.toString(),
        error: (error as Error).message
      });
    }
  }

  /**
   * Handle heartbeat response
   */
  private handleHeartbeatResponse(response: HeartbeatResponse, followerAddress: Address): void {
    if (response.term > this.currentTerm) {
      this.becomeFollower(response.term);
      return;
    }

    if (response.success) {
      this.logger.heartbeat('Heartbeat acknowledged', {
        from: followerAddress.toString()
      });
      
      if (response.matchIndex !== undefined) {
        this.matchIndex.set(followerAddress.hashCode(), response.matchIndex);
      }
    } else {
      this.logger.warn('Heartbeat rejected', {
        from: followerAddress.toString(),
        term: response.term
      });
    }
  }

  /**
   * Start election timeout
   */
  private startElectionTimeout(): void {
    if (this.electionTimer) {
      clearTimeout(this.electionTimer);
    }

    const timeout = Math.random() * 
      (RaftNode.ELECTION_TIMEOUT_MAX - RaftNode.ELECTION_TIMEOUT_MIN) + 
      RaftNode.ELECTION_TIMEOUT_MIN;

    this.electionTimer = setTimeout(() => {
      if (this.nodeType === NodeType.FOLLOWER && this.running) {
        const timeSinceLastHeartbeat = Date.now() - this.lastHeartbeatTime;
        
        if (timeSinceLastHeartbeat >= RaftNode.ELECTION_TIMEOUT_MIN) {
          this.logger.election('Election timeout - starting election');
          this.startElection();
        } else {
          this.startElectionTimeout();
        }
      }
    }, timeout);
  }

  /**
   * Start election process
   */
  private async startElection(): Promise<void> {
    if (!this.running || this.nodeType === NodeType.LEADER) {
      return;
    }

    this.logger.election('Starting election', { term: this.currentTerm + 1 });

    this.nodeType = NodeType.CANDIDATE;
    this.currentTerm++;
    this.votedFor = this.address;
    this.votesReceived.clear();
    this.votesReceived.add(this.address.hashCode());

    this.startElectionTimeout();
    await this.requestVotes();
  }

  /**
   * Request votes from all other nodes
   */
  private async requestVotes(): Promise<void> {
    const voteRequest: VoteRequest = {
      term: this.currentTerm,
      candidateId: this.address.toObject(),
      lastLogIndex: this.log.length - 1,
      lastLogTerm: this.log.length > 0 ? this.log[this.log.length - 1]!.term : 0
    };

    const votePromises: Promise<void>[] = [];

    for (const addr of this.clusterAddresses) {
      if (!addr.equals(this.address)) {
        votePromises.push(this.requestVoteFromNode(addr, voteRequest));
      }
    }

    try {
      await Promise.race([
        Promise.allSettled(votePromises),
        new Promise(resolve => setTimeout(resolve, 2000))
      ]);
    } catch (error) {
      this.logger.warn('Error during vote collection', { error: (error as Error).message });
    }

    this.checkElectionResult();
  }

  /**
   * Request vote from specific node
   */
  private async requestVoteFromNode(address: Address, request: VoteRequest): Promise<void> {
    try {
      const response = await this.sendRPC<VoteResponse>(address, 'requestVote', request);
      this.handleVoteResponse(response, address);
    } catch (error) {
      this.logger.warn('Failed to request vote', { 
        target: address.toString(),
        error: (error as Error).message 
      });
    }
  }

  /**
   * Handle vote response
   */
  private handleVoteResponse(response: VoteResponse, voterAddress: Address): void {
    if (this.nodeType !== NodeType.CANDIDATE) {
      return;
    }

    if (response.term > this.currentTerm) {
      this.becomeFollower(response.term);
      return;
    }

    if (response.term === this.currentTerm && response.voteGranted) {
      this.votesReceived.add(voterAddress.hashCode());
      
      const majority = Math.floor(this.clusterAddresses.size / 2) + 1;
      if (this.votesReceived.size >= majority) {
        this.becomeLeader();
      }
    }
  }

  /**
   * Check election result
   */
  private checkElectionResult(): void {
    if (this.nodeType !== NodeType.CANDIDATE) {
      return;
    }

    const majority = Math.floor(this.clusterAddresses.size / 2) + 1;
    
    if (this.votesReceived.size >= majority) {
      this.becomeLeader();
    } else {
      this.becomeFollower(this.currentTerm);
    }
  }

  /**
   * Replicate to follower with retry
   */
  private async replicateToFollowerWithRetry(follower: Address, entries: LogEntry[], maxRetries: number = 2): Promise<boolean> {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const success = await this.replicateToFollower(follower, entries);
        if (success) {
          return true;
        }
        
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
        }
      } catch (error) {
        if (attempt === maxRetries) {
          return false;
        }
      }
    }
    return false;
  }

  /**
   * Replicate entries to specific follower
   */
  private async replicateToFollower(follower: Address, entries: LogEntry[]): Promise<boolean> {
    const prevLogIndex = entries[0]!.index - 1;
    const prevLogTerm = prevLogIndex >= 0 ? this.log[prevLogIndex]?.term || 0 : 0;

    const request: LogReplicationRequest = {
      term: this.currentTerm,
      leaderId: this.address.toObject(),
      prevLogIndex,
      prevLogTerm,
      entries,
      leaderCommit: this.commitIndex
    };

    try {
      const response = await this.sendRPC<LogReplicationResponse>(
        follower, 
        'logReplication', 
        request
      );

      if (response.term > this.currentTerm) {
        this.becomeFollower(response.term);
        return false;
      }

      if (response.success) {
        this.matchIndex.set(follower.hashCode(), response.matchIndex);
        this.nextIndex.set(follower.hashCode(), response.matchIndex + 1);
        return true;
      } else {
        await this.handleLogInconsistency(follower, response);
        return false;
      }

    } catch (error) {
      this.logger.warn('Replication to follower failed', {
        follower: follower.toString(),
        error: (error as Error).message
      });
      return false;
    }
  }

  /**
   * Handle log inconsistency with follower
   */
  private async handleLogInconsistency(follower: Address, response: LogReplicationResponse): Promise<void> {
    const currentNext = this.nextIndex.get(follower.hashCode()) || this.log.length;
    this.nextIndex.set(follower.hashCode(), Math.max(0, currentNext - 1));
  }

  /**
   * Generate unique entry ID
   */
  private generateEntryId(): string {
    return `${this.address.hashCode()}-${this.currentTerm}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle cluster state synchronization (for Server.ts)
   */
  async handleClusterStateSync(clusterState: {
    members: AddressData[];
    leader: AddressData;
    term: number;
    log: LogEntry[];
  }): Promise<{ success: boolean; message?: string }> {
    this.logger.membership('Received cluster state sync', {
      newClusterSize: clusterState.members?.length,
      newTerm: clusterState.term,
      logEntries: clusterState.log?.length
    });

    try {
      // Update cluster membership
      if (clusterState.members) {
        this.syncClusterState(clusterState.members, clusterState.term);
      }

      // Update leader
      if (clusterState.leader) {
        this.leaderAddress = Address.fromObject(clusterState.leader);
      }

      // Update term
      if (clusterState.term > this.currentTerm) {
        this.currentTerm = clusterState.term;
        if (this.nodeType !== NodeType.FOLLOWER) {
          this.becomeFollower(clusterState.term, this.leaderAddress ?? undefined);
        }
      }

      // Sync log if provided and our log is behind
      if (clusterState.log && Array.isArray(clusterState.log)) {
        if (clusterState.log.length > this.log.length) {
          this.log = clusterState.log;
          this.commitIndex = Math.min(this.commitIndex, this.log.length - 1);
          await this.applyLogEntries();
        }
      }

      return { success: true, message: 'Cluster state synchronized successfully' };

    } catch (error) {
      this.logger.error('Failed to sync cluster state', error as Error);
      return { success: false, message: (error as Error).message };
    }
  }

  /**
   * Handle removal notification (for Server.ts RPC)
   */
  async handleRemovalNotification(params: {
    reason: string;
    term: number;
    removedBy: AddressData;
    timestamp: number;
  }): Promise<{ acknowledged: boolean; message: string }> {
    this.logger.membership('Received removal notification', {
      reason: params.reason,
      term: params.term,
      removedBy: `${params.removedBy.ip}:${params.removedBy.port}`
    });

    try {
      // Acknowledge the removal and prepare for graceful shutdown
      setTimeout(async () => {
        this.logger.info('Gracefully shutting down due to removal from cluster');
        await this.stop();
        process.exit(0);
      }, 5000);

      return {
        acknowledged: true,
        message: 'Removal acknowledged, shutting down gracefully in 5 seconds'
      };

    } catch (error) {
      this.logger.error('Failed to handle removal notification', error as Error);
      return {
        acknowledged: false,
        message: (error as Error).message
      };
    }
  }

  /**
   * Add node to cluster (for Server.ts membership/add endpoint)
   */
  async addNodeToCluster(nodeAddress: Address): Promise<boolean> {
    this.logger.membership('addNodeToCluster called', {
      nodeAddress: nodeAddress.toString(),
      isLeader: this.nodeType === NodeType.LEADER,
      clusterSize: this.clusterAddresses.size
    });

    // Only leader can add nodes
    if (this.nodeType !== NodeType.LEADER) {
      this.logger.warn('Cannot add node: not the leader', {
        nodeAddress: nodeAddress.toString(),
        currentRole: this.nodeType
      });
      return false;
    }

    try {
      // Check if node already exists
      let alreadyExists = false;
      for (const addr of this.clusterAddresses) {
        if (addr.equals(nodeAddress)) {
          alreadyExists = true;
          break;
        }
      }

      if (alreadyExists) {
        this.logger.membership('Node already in cluster', {
          nodeAddress: nodeAddress.toString()
        });
        return true; // Consider this a success since the end result is achieved
      }

      // Test connectivity to new node
      const isReachable = await this.testNodeConnectivity(nodeAddress);
      if (!isReachable) {
        this.logger.warn('Cannot reach new node', {
          nodeAddress: nodeAddress.toString()
        });
        return false;
      }

      // Add to cluster state
      this.addMemberToCluster(nodeAddress);

      // Initialize leader state for new node
      if (this.nodeType === NodeType.LEADER) {
        this.nextIndex.set(nodeAddress.hashCode(), this.log.length);
        this.matchIndex.set(nodeAddress.hashCode(), -1);
      }

      // Broadcast membership change to existing nodes
      await this.broadcastMembershipChange('add', nodeAddress);

      // Send current cluster state to new node
      await this.sendClusterStateToNode(nodeAddress);

      this.logger.membership('Node successfully added to cluster', {
        nodeAddress: nodeAddress.toString(),
        newClusterSize: this.clusterAddresses.size
      });

      this.emit('memberAdded', { address: nodeAddress, clusterSize: this.clusterAddresses.size });
      return true;

    } catch (error) {
      this.logger.error('Failed to add node to cluster', error as Error, {
        nodeAddress: nodeAddress.toString()
      });
      return false;
    }
  }

  /**
   * Test connectivity to a node
   */
  private async testNodeConnectivity(nodeAddress: Address): Promise<boolean> {
    try {
      const response = await this.sendRPC(
        nodeAddress,
        'ping',
        {},
        { timeout: 3000 }
      );
      
      return (response as { pong?: boolean }).pong === true;
    } catch (error) {
      this.logger.debug('Node connectivity test failed', {
        nodeAddress: nodeAddress.toString(),
        errorMessage: (error as Error).message
      });
      return false;
    }
  }

  /**
   * Send cluster state to specific node
   */
  private async sendClusterStateToNode(nodeAddress: Address): Promise<void> {
    const clusterState = {
      members: Array.from(this.clusterAddresses).map(addr => addr.toObject()),
      leader: this.address.toObject(),
      term: this.currentTerm,
      log: this.log // Send current log for synchronization
    };

    try {
      await this.sendRPC(
        nodeAddress,
        'syncClusterState',
        clusterState
      );

      this.logger.membership('Cluster state sent to new node', {
        nodeAddress: nodeAddress.toString(),
        clusterSize: clusterState.members.length,
        logEntries: clusterState.log.length
      });

    } catch (error) {
      this.logger.warn('Failed to send cluster state to node', {
        nodeAddress: nodeAddress.toString(),
        error: (error as Error).message
      });
    }
  }

  /**
   * Send RPC request to another node with options
   */
  private async sendRPC<T>(address: Address, method: string, params: any, options?: { timeout?: number }): Promise<T> {
    if (!this.rpcClient) {
      throw new Error('RPC client not initialized');
    }

    try {
      const result = await this.rpcClient.call(address.toURL(), method, params, options);
      return result as T;
    } catch (error) {
      throw new Error(`RPC call failed: ${(error as Error).message}`);
    }
  }
}