/**
 * Dedicated Log Replication Manager (Farhan's Implementation)
 * Handles complex log replication logic separate from main RaftNode
 */

import { Address } from './Address';
import { Logger } from '../utils/Logger';
import { LogEntry, LogReplicationRequest, LogReplicationResponse } from './RaftNode';

export interface ReplicationState {
  nextIndex: number;
  matchIndex: number;
  isReplicating: boolean;
  lastAttempt: number;
  consecutiveFailures: number;
}

export interface ReplicationResult {
  success: boolean;
  matchIndex: number;
  needsRetry: boolean;
  error?: string;
}

export class LogReplicationManager {
  private logger: Logger;
  private rpcClient: any;
  private replicationStates: Map<string, ReplicationState> = new Map();
  
  // Configuration
  private readonly MAX_CONSECUTIVE_FAILURES = 5;
  private readonly REPLICATION_RETRY_DELAY = 100; // ms
  private readonly BATCH_SIZE = 10; // Max entries per replication request

  constructor(logger: Logger, rpcClient: any) {
    this.logger = logger;
    this.rpcClient = rpcClient;
  }

  /**
   * Initialize replication state for a follower
   */
  initializeFollower(followerAddr: Address, logLength: number): void {
    const state: ReplicationState = {
      nextIndex: logLength,
      matchIndex: -1,
      isReplicating: false,
      lastAttempt: 0,
      consecutiveFailures: 0
    };

    this.replicationStates.set(followerAddr.hashCode(), state);
    
    this.logger.logReplication('Initialized replication state for follower', {
      follower: followerAddr.toString(),
      nextIndex: state.nextIndex,
      matchIndex: state.matchIndex
    });
  }

  /**
   * Remove follower from replication
   */
  removeFollower(followerAddr: Address): void {
    this.replicationStates.delete(followerAddr.hashCode());
    this.logger.logReplication('Removed follower from replication', {
      follower: followerAddr.toString()
    });
  }

  /**
   * Replicate entries to a specific follower
   */
  async replicateToFollower(
    followerAddr: Address,
    log: LogEntry[],
    currentTerm: number,
    leaderAddr: Address,
    commitIndex: number
  ): Promise<ReplicationResult> {
    const followerKey = followerAddr.hashCode();
    const state = this.replicationStates.get(followerKey);

    if (!state) {
      throw new Error(`No replication state for follower ${followerAddr.toString()}`);
    }

    if (state.isReplicating) {
      return {
        success: false,
        matchIndex: state.matchIndex,
        needsRetry: true,
        error: 'Replication already in progress'
      };
    }

    // Check if follower has too many consecutive failures
    if (state.consecutiveFailures >= this.MAX_CONSECUTIVE_FAILURES) {
      const timeSinceLastAttempt = Date.now() - state.lastAttempt;
      if (timeSinceLastAttempt < this.REPLICATION_RETRY_DELAY * state.consecutiveFailures) {
        return {
          success: false,
          matchIndex: state.matchIndex,
          needsRetry: true,
          error: 'Follower temporarily unavailable'
        };
      }
    }

    state.isReplicating = true;
    state.lastAttempt = Date.now();

    try {
      // Determine entries to send
      const nextIndex = state.nextIndex;
      const entriesToSend = this.getEntriesToSend(log, nextIndex);

      // Calculate previous log entry info
      const prevLogIndex = nextIndex - 1;
      const prevLogTerm = prevLogIndex >= 0 && prevLogIndex < log.length 
        ? log[prevLogIndex]!.term 
        : 0;

      const request: LogReplicationRequest = {
        term: currentTerm,
        leaderId: leaderAddr.toObject(),
        prevLogIndex,
        prevLogTerm,
        entries: entriesToSend,
        leaderCommit: commitIndex
      };

      this.logger.logReplication('Sending replication request', {
        follower: followerAddr.toString(),
        prevLogIndex,
        prevLogTerm,
        entriesCount: entriesToSend.length,
        nextIndex: state.nextIndex
      });

      const response = await this.rpcClient.call(
        followerAddr.toURL(),
        'logReplication',
        request,
        { timeout: 2000 }
      );

      return this.handleReplicationResponse(followerKey, response, entriesToSend.length);

    } catch (error) {
      this.logger.warn('Replication request failed', {
        follower: followerAddr.toString(),
        error: (error as Error).message,
        consecutiveFailures: state.consecutiveFailures + 1
      });

      state.consecutiveFailures++;
      return {
        success: false,
        matchIndex: state.matchIndex,
        needsRetry: true,
        error: (error as Error).message
      };

    } finally {
      state.isReplicating = false;
    }
  }

  /**
   * Handle replication response from follower
   */
  private handleReplicationResponse(
    followerKey: string,
    response: LogReplicationResponse,
    entriesSent: number
  ): ReplicationResult {
    const state = this.replicationStates.get(followerKey);
    if (!state) {
      return {
        success: false,
        matchIndex: -1,
        needsRetry: false,
        error: 'Lost replication state'
      };
    }

    if (response.success) {
      // Successful replication
      state.matchIndex = response.matchIndex;
      state.nextIndex = response.matchIndex + 1;
      state.consecutiveFailures = 0;

      this.logger.logReplication('Replication successful', {
        followerKey,
        matchIndex: state.matchIndex,
        nextIndex: state.nextIndex,
        entriesSent
      });

      return {
        success: true,
        matchIndex: state.matchIndex,
        needsRetry: false
      };

    } else {
      // Failed replication - handle log inconsistency
      this.handleLogInconsistency(state, response);
      state.consecutiveFailures++;

      return {
        success: false,
        matchIndex: state.matchIndex,
        needsRetry: true,
        error: 'Log inconsistency detected'
      };
    }
  }

  /**
   * Handle log inconsistency with follower
   */
  private handleLogInconsistency(state: ReplicationState, response: LogReplicationResponse): void {
    if (response.conflictIndex !== undefined) {
      // Use conflict information to optimize nextIndex
      state.nextIndex = Math.max(0, response.conflictIndex);
    } else {
      // Fallback: decrement nextIndex
      state.nextIndex = Math.max(0, state.nextIndex - 1);
    }

    this.logger.logReplication('Handling log inconsistency', {
      oldNextIndex: state.nextIndex + (response.conflictIndex !== undefined ? response.conflictIndex - state.nextIndex : 1),
      newNextIndex: state.nextIndex,
      conflictIndex: response.conflictIndex,
      conflictTerm: response.conflictTerm
    });
  }

  /**
   * Get entries to send to follower
   */
  private getEntriesToSend(log: LogEntry[], nextIndex: number): LogEntry[] {
    if (nextIndex >= log.length) {
      return []; // Follower is up to date
    }

    const startIndex = nextIndex;
    const endIndex = Math.min(startIndex + this.BATCH_SIZE, log.length);

    return log.slice(startIndex, endIndex);
  }

  /**
   * Replicate to multiple followers concurrently
   */
  async replicateToFollowers(
    followers: Address[],
    log: LogEntry[],
    currentTerm: number,
    leaderAddr: Address,
    commitIndex: number
  ): Promise<Map<string, ReplicationResult>> {
    const replicationPromises = followers.map(async (follower) => {
      const result = await this.replicateToFollower(
        follower,
        log,
        currentTerm,
        leaderAddr,
        commitIndex
      );
      return { follower: follower.hashCode(), result };
    });

    const results = await Promise.allSettled(replicationPromises);
    const replicationResults = new Map<string, ReplicationResult>();

    results.forEach((promiseResult, index) => {
      const follower = followers[index]!;
      
      if (promiseResult.status === 'fulfilled') {
        replicationResults.set(
          promiseResult.value.follower,
          promiseResult.value.result
        );
      } else {
        // Promise rejected
        replicationResults.set(follower.hashCode(), {
          success: false,
          matchIndex: -1,
          needsRetry: true,
          error: promiseResult.reason?.message || 'Unknown error'
        });
      }
    });

    return replicationResults;
  }

  /**
   * Check if majority of followers have replicated up to given index
   */
  checkMajorityReplication(targetIndex: number, clusterSize: number): boolean {
    const replicatedCount = Array.from(this.replicationStates.values())
      .filter(state => state.matchIndex >= targetIndex)
      .length + 1; // +1 for leader

    const majorityThreshold = Math.floor(clusterSize / 2) + 1;
    
    this.logger.logReplication('Checking majority replication', {
      targetIndex,
      replicatedCount,
      majorityThreshold,
      clusterSize,
      hasMajority: replicatedCount >= majorityThreshold
    });

    return replicatedCount >= majorityThreshold;
  }

  /**
   * Get replication statistics
   */
  getReplicationStats(): {
    totalFollowers: number;
    activeReplications: number;
    averageMatchIndex: number;
    followersWithFailures: number;
    replicationHealth: 'healthy' | 'degraded' | 'critical';
  } {
    const states = Array.from(this.replicationStates.values());
    const totalFollowers = states.length;
    const activeReplications = states.filter(s => s.isReplicating).length;
    const averageMatchIndex = totalFollowers > 0 
      ? states.reduce((sum, s) => sum + s.matchIndex, 0) / totalFollowers 
      : -1;
    const followersWithFailures = states.filter(s => s.consecutiveFailures > 0).length;

    let replicationHealth: 'healthy' | 'degraded' | 'critical';
    if (followersWithFailures === 0) {
      replicationHealth = 'healthy';
    } else if (followersWithFailures < totalFollowers / 2) {
      replicationHealth = 'degraded';
    } else {
      replicationHealth = 'critical';
    }

    return {
      totalFollowers,
      activeReplications,
      averageMatchIndex,
      followersWithFailures,
      replicationHealth
    };
  }

  /**
   * Get detailed replication state for debugging
   */
  getDetailedState(): Array<{
    follower: string;
    nextIndex: number;
    matchIndex: number;
    isReplicating: boolean;
    consecutiveFailures: number;
    lastAttempt: number;
  }> {
    const result: Array<any> = [];
    
    this.replicationStates.forEach((state, followerKey) => {
      result.push({
        follower: followerKey,
        nextIndex: state.nextIndex,
        matchIndex: state.matchIndex,
        isReplicating: state.isReplicating,
        consecutiveFailures: state.consecutiveFailures,
        lastAttempt: state.lastAttempt
      });
    });

    return result;
  }

  /**
   * Reset replication state for all followers
   */
  resetAllStates(logLength: number): void {
    this.replicationStates.forEach((state) => {
      state.nextIndex = logLength;
      state.matchIndex = -1;
      state.isReplicating = false;
      state.consecutiveFailures = 0;
      state.lastAttempt = 0;
    });

    this.logger.logReplication('Reset all replication states', {
      followerCount: this.replicationStates.size,
      newNextIndex: logLength
    });
  }

  /**
   * Update configuration
   */
  updateConfig(config: {
    maxConsecutiveFailures?: number;
    retryDelay?: number;
    batchSize?: number;
  }): void {
    if (config.maxConsecutiveFailures !== undefined) {
      (this as any).MAX_CONSECUTIVE_FAILURES = config.maxConsecutiveFailures;
    }
    if (config.retryDelay !== undefined) {
      (this as any).REPLICATION_RETRY_DELAY = config.retryDelay;
    }
    if (config.batchSize !== undefined) {
      (this as any).BATCH_SIZE = config.batchSize;
    }

    this.logger.info('Updated replication manager configuration', config);
  }
}