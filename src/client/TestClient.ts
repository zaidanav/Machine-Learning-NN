/**
 * Enhanced Test client for Raft cluster with Log Replication Testing
 * TypeScript implementation for demo and comprehensive testing
 * Enhanced by Farhan for Log Replication testing scenarios
 */

import { Address } from '../core/Address';
import { Logger } from '../utils/Logger';

// Import fetch for Node.js
const fetch = require('node-fetch');
const readline = require('readline');

interface NodeInfo {
  address: Address;
  type: string;
  currentTerm: number;
  leader?: Address;
  healthy: boolean;
}

interface CommandResult {
  success: boolean;
  result?: unknown;
  error?: string;
  redirect?: string;
  leader?: string;
  timestamp?: number;
}

interface LogResponse {
  success: boolean;
  log: LogEntry[];
  metadata: {
    commitIndex: number;
    lastApplied: number;
    logLength: number;
    currentTerm: number;
    clusterSize: number;
  };
  clusterInfo: {
    address: { ip: string; port: number };
    type: string;
    currentTerm: number;
    clusterSize: number;
    leader?: { ip: string; port: number };
  };
  timestamp: number;
}

interface LogEntry {
  term: number;
  index: number;
  command: string;
  key?: string;
  value?: string;
  timestamp: string;
  id: string;
}

interface LogStats {
  success: boolean;
  stats: {
    totalEntries: number;
    commitIndex: number;
    lastLogIndex: number;
    termDistribution: Record<number, number>;
    commandTypes: Record<string, number>;
    recentEntries: Array<{
      index: number;
      term: number;
      command: string;
      timestamp: string;
    }>;
  };
  timestamp: number;
}

interface HealthResponse {
  status: string;
  nodeId: string;
  address: string;
  raftStatus: {
    type: string;
    currentTerm: number;
    clusterSize: number;
    leader?: { ip: string; port: number };
  };
  timestamp: string;
}

export class RaftTestClient {
  private nodes: Address[] = [];
  private currentLeader: Address | null = null;
  private logger: Logger;
  private timeout: number = 5000;

  constructor(nodes?: Address[]) {
    // Default Docker network addresses
    this.nodes = nodes || [
      new Address('172.20.0.10', 8001), // raft-node-1
      new Address('172.20.0.11', 8001), // raft-node-2
      new Address('172.20.0.12', 8001), // raft-node-3
      new Address('172.20.0.13', 8001), // raft-node-4
    ];

    this.logger = new Logger({ nodeId: 'test-client' });
    this.logger.info('Test client initialized with log replication testing', {
      nodes: this.nodes.map(n => n.toString())
    });
  }

  /**
   * Find current leader in the cluster
   */
  async findLeader(): Promise<Address | null> {
    this.logger.info('🔍 Searching for cluster leader...');

    for (const node of this.nodes) {
      try {
        const response = await fetch(`http://${node.toString()}/cluster_info`, {
          timeout: this.timeout
        });

        if (!response.ok) {
          console.log(`   ❌ ${node.toString()} - HTTP ${response.status}`);
          continue;
        }

        const info = await response.json();
        
        if (info.type === 'LEADER') {
          this.currentLeader = node;
          console.log(`   ✅ ${node.toString()} - LEADER (Term: ${info.currentTerm})`);
          return node;
        } else {
          const leaderInfo = info.leader;
          if (leaderInfo) {
            const leader = new Address(leaderInfo.ip, leaderInfo.port);
            console.log(`   📍 ${node.toString()} - ${info.type} (Leader: ${leader.toString()})`);
            this.currentLeader = leader;
            return leader;
          } else {
            console.log(`   ⚠️  ${node.toString()} - ${info.type} (No leader known)`);
          }
        }

      } catch (error) {
        console.log(`   ❌ ${node.toString()} - ${(error as Error).message}`);
      }
    }

    console.log('   🚫 No leader found in cluster!');
    return null;
  }

  /**
   * Get status of all nodes
   */
  async getClusterStatus(): Promise<void> {
    console.log('\n' + '='.repeat(60));
    console.log('                CLUSTER STATUS');
    console.log('='.repeat(60));

    let leaderCount = 0;
    let followerCount = 0;
    let candidateCount = 0;
    let offlineCount = 0;

    for (const node of this.nodes) {
      try {
        const response = await fetch(`http://${node.toString()}/health`, {
          timeout: this.timeout
        });

        if (!response.ok) {
          console.log(`🔴 ${node.toString()} - HTTP ${response.status}`);
          offlineCount++;
          continue;
        }

        const health: HealthResponse = await response.json();
        const raftStatus = health.raftStatus;
        
        let icon = '❓';
        if (raftStatus.type === 'LEADER') {
          icon = '👑';
          leaderCount++;
        } else if (raftStatus.type === 'FOLLOWER') {
          icon = '👥';
          followerCount++;
        } else if (raftStatus.type === 'CANDIDATE') {
          icon = '🗳️';
          candidateCount++;
        }

        console.log(`${icon} ${node.toString()} - ${raftStatus.type} (Term: ${raftStatus.currentTerm}, Cluster: ${raftStatus.clusterSize})`);

      } catch (error) {
        console.log(`🔴 ${node.toString()} - OFFLINE`);
        offlineCount++;
      }
    }

    console.log('-'.repeat(60));
    console.log(`Summary: ${leaderCount} Leader(s), ${followerCount} Follower(s), ${candidateCount} Candidate(s), ${offlineCount} Offline`);
    console.log('='.repeat(60));
  }

  /**
   * Execute command on leader with enhanced error handling
   */
  async executeCommand(command: string, targetNode?: Address): Promise<CommandResult> {
    const target = targetNode || await this.findLeader();
    
    if (!target) {
      return {
        success: false,
        error: 'No leader available'
      };
    }

    try {
      const response = await fetch(`http://${target.toString()}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ command }),
        timeout: this.timeout
      });

      const result: CommandResult = await response.json();

      if (!response.ok) {
        // Check for redirect
        if (result.redirect) {
          console.log(`   🔄 Redirected to: ${result.leader}`);
          return result;
        }
        return {
          success: false,
          error: result.error || `HTTP ${response.status}`
        };
      }

      return result;

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Get log from leader with enhanced response handling
   */
  async requestLog(targetNode?: Address): Promise<LogResponse | null> {
    const target = targetNode || await this.findLeader();
    
    if (!target) {
      console.log('❌ No leader available for log request');
      return null;
    }

    try {
      const response = await fetch(`http://${target.toString()}/request_log`, {
        timeout: this.timeout
      });

      if (!response.ok) {
        const error = await response.json();
        console.log(`❌ Log request failed: ${error.error}`);
        if (error.redirect) {
          console.log(`   🔄 Try: ${error.redirect}`);
        }
        return null;
      }

      const result: LogResponse = await response.json();
      return result;

    } catch (error) {
      console.log(`❌ Error requesting log: ${(error as Error).message}`);
      return null;
    }
  }

  /**
   * Get log statistics from leader
   */
  async getLogStats(targetNode?: Address): Promise<LogStats | null> {
    const target = targetNode || await this.findLeader();
    
    if (!target) {
      console.log('❌ No leader available for log stats');
      return null;
    }

    try {
      const response = await fetch(`http://${target.toString()}/log_stats`, {
        timeout: this.timeout
      });

      if (!response.ok) {
        const error = await response.json();
        console.log(`❌ Log stats request failed: ${error.error}`);
        return null;
      }

      const result: LogStats = await response.json();
      return result;

    } catch (error) {
      console.log(`❌ Error requesting log stats: ${(error as Error).message}`);
      return null;
    }
  }

  /**
   * Test heartbeat by monitoring cluster for duration
   */
  async testHeartbeat(duration: number = 10): Promise<void> {
    console.log(`\n🫀 HEARTBEAT TEST (${duration} seconds)`);
    console.log('-'.repeat(50));

    const startTime = Date.now();
    while (Date.now() - startTime < duration * 1000) {
      await this.getClusterStatus();
      console.log();
      await this.sleep(2000);
    }

    console.log('✅ Heartbeat test completed!');
  }

  /**
   * Test ping connectivity
   */
  async testPing(): Promise<void> {
    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ Cannot test ping - no leader found');
      return;
    }

    const result = await this.executeCommand('ping');
    if (result.success) {
      console.log(`🏓 Ping to leader ${leader.toString()}: ${result.result}`);
    } else {
      console.log(`❌ Ping failed: ${result.error}`);
    }
  }

  /**
   * Test log replication with detailed verification (Farhan's Implementation)
   */
  async testLogReplication(): Promise<void> {
    console.log('\n📋 LOG REPLICATION TEST');
    console.log('-'.repeat(50));

    // Find leader
    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ No leader found for log replication test');
      return;
    }

    console.log(`📊 Testing log replication with leader: ${leader.toString()}`);

    // Test commands from assignment specification
    const testCommands = [
      'set "1" "A"',
      'append "1" "BC"', 
      'set "2" "SI"',
      'append "2" "S"',
      'get "1"'
    ];

    console.log('\n1️⃣ Executing commands on leader...');
    for (const cmd of testCommands) {
      console.log(`   Executing: ${cmd}`);
      const result = await this.executeCommand(cmd);
      
      if (result.success) {
        console.log(`   ✅ Result: ${result.result}`);
      } else {
        console.log(`   ❌ Failed: ${result.error}`);
      }
      
      await this.sleep(1000); // Wait for replication
    }

    // Test commands on non-leader nodes
    console.log('\n2️⃣ Testing commands on follower nodes...');
    const followers = this.nodes.filter(node => !node.equals(leader));
    
    for (const follower of followers.slice(0, 2)) { // Test on first 2 followers
      console.log(`\n   Testing follower: ${follower.toString()}`);
      
      const getResult1 = await this.executeCommand('get "1"', follower);
      const getResult2 = await this.executeCommand('get "2"', follower);
      
      if (getResult1.redirect) {
        console.log(`   🔄 Redirected to leader: ${getResult1.redirect}`);
        // Try the redirect
        const redirectResult = await this.executeCommand('get "1"');
        console.log(`   ✅ Redirect result: ${redirectResult.result}`);
      } else {
        console.log(`   📖 Direct result: ${getResult1.result}`);
      }
    }

    // Verify log consistency
    console.log('\n3️⃣ Verifying log consistency...');
    const logResponse = await this.requestLog();
    if (logResponse) {
      console.log(`   📊 Log entries: ${logResponse.log.length}`);
      console.log(`   ✅ Commit index: ${logResponse.metadata.commitIndex}`);
      console.log(`   📈 Last applied: ${logResponse.metadata.lastApplied}`);
    }
  }

  /**
   * Test concurrent commands from different clients (Farhan's Implementation)
   */
  async testConcurrentCommands(): Promise<void> {
    console.log('\n🔄 CONCURRENT COMMANDS TEST');
    console.log('-'.repeat(50));

    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ No leader found');
      return;
    }

    console.log('Testing concurrent commands as specified in assignment...');

    // Commands for Node 1 (simulated)
    const node1Commands = [
      'set "ruby-chan" "choco-minto"',
      'append "ruby-chan" "-yori-mo-anata"'
    ];

    // Commands for Node 2 (simulated)  
    const node2Commands = [
      'set "ayumu-chan" "strawberry-flavor"',
      'append "ayumu-chan" "-yori-mo-anata"'
    ];

    console.log('\n🚀 Executing concurrent commands...');
    
    // Execute commands concurrently
    const promises = [
      ...node1Commands.map(cmd => this.executeCommand(cmd)),
      ...node2Commands.map(cmd => this.executeCommand(cmd))
    ];

    try {
      const results = await Promise.allSettled(promises);
      
      results.forEach((result, index) => {
        const allCommands = [...node1Commands, ...node2Commands];
        const cmd = allCommands[index];
        
        if (result.status === 'fulfilled' && result.value.success) {
          console.log(`   ✅ ${cmd}: ${result.value.result}`);
        } else {
          const error = result.status === 'rejected' 
            ? result.reason.message 
            : result.value.error;
          console.log(`   ❌ ${cmd}: ${error}`);
        }
      });

    } catch (error) {
      console.log(`❌ Concurrent execution failed: ${(error as Error).message}`);
    }

    // Verify final state
    console.log('\n🔍 Verifying final state...');
    await this.sleep(2000); // Wait for replication

    const finalResults = await Promise.allSettled([
      this.executeCommand('get "ruby-chan"'),
      this.executeCommand('get "ayumu-chan"')
    ]);

    finalResults.forEach((result, index) => {
      const keys = ['ruby-chan', 'ayumu-chan'];
      const key = keys[index];
      
      if (result.status === 'fulfilled' && result.value.success) {
        console.log(`   📊 Final ${key}: "${result.value.result}"`);
      } else {
        console.log(`   ❌ Failed to get ${key}`);
      }
    });
  }

  /**
   * Test log consistency across cluster (Farhan's Implementation)
   */
  async testLogConsistency(): Promise<void> {
    console.log('\n🔍 LOG CONSISTENCY TEST');
    console.log('-'.repeat(50));

    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ No leader found');
      return;
    }

    // Get log from leader
    console.log('📋 Requesting log from leader...');
    const leaderLog = await this.requestLog(leader);
    
    if (!leaderLog) {
      console.log('❌ Failed to get leader log');
      return;
    }

    console.log(`📊 Leader log contains ${leaderLog.log.length} entries`);
    console.log(`📈 Commit index: ${leaderLog.metadata.commitIndex}`);
    console.log(`🔄 Last applied: ${leaderLog.metadata.lastApplied}`);
    console.log(`📊 Current term: ${leaderLog.metadata.currentTerm}`);

    // Show recent log entries
    if (leaderLog.log.length > 0) {
      console.log('\n📝 Recent log entries:');
      const recentEntries = leaderLog.log.slice(-5); // Last 5 entries
      
      recentEntries.forEach((entry: LogEntry) => {
        const keyValue = entry.key ? `${entry.key}=${entry.value}` : '';
        console.log(`   [${entry.index}] Term:${entry.term} ${entry.command} ${keyValue}`);
      });
    }

    // Test log statistics endpoint
    try {
      console.log('\n📊 Requesting log statistics...');
      const stats = await this.getLogStats(leader);
      
      if (stats) {
        console.log(`   Total entries: ${stats.stats.totalEntries}`);
        console.log(`   Command types: ${JSON.stringify(stats.stats.commandTypes)}`);
        console.log(`   Term distribution: ${JSON.stringify(stats.stats.termDistribution)}`);
      }
    } catch (error) {
      console.log('   ⚠️ Could not get log statistics');
    }
  }

  /**
   * Test leader election monitoring (Enhanced)
   */
  async testLeaderElection(): Promise<void> {
    console.log('\n🗳️  LEADER ELECTION TEST');
    console.log('-'.repeat(50));

    // Find current leader
    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ No leader found for election test');
      return;
    }

    console.log(`📊 Current leader: ${leader.toString()}`);
    
    // Get initial state
    const initialLog = await this.requestLog();
    console.log(`📋 Initial log entries: ${initialLog?.log.length || 0}`);
    
    console.log('⚠️  To test election, manually stop the leader container:');
    
    // Find which container number this leader corresponds to
    const leaderIndex = this.nodes.findIndex(n => n.equals(leader));
    if (leaderIndex >= 0) {
      console.log(`   docker stop raft-node-${leaderIndex + 1}`);
    }
    
    console.log();
    console.log('🕐 Monitoring cluster for 30 seconds...');

    // Monitor for 30 seconds
    for (let i = 0; i < 15; i++) {
      await this.sleep(2000);
      console.log(`\n--- Check ${i + 1}/15 ---`);
      await this.getClusterStatus();

      // Check if new leader emerged
      const newLeader = await this.findLeader();
      if (newLeader && !newLeader.equals(leader)) {
        console.log(`🎉 New leader elected: ${newLeader.toString()}`);
        
        // Test log preservation
        const newLeaderLog = await this.requestLog(newLeader);
        if (newLeaderLog && initialLog) {
          const preserved = newLeaderLog.log.length >= initialLog.log.length;
          console.log(`📊 Log preserved: ${preserved ? '✅' : '❌'}`);
          console.log(`📈 Entries: ${initialLog.log.length} → ${newLeaderLog.log.length}`);
        }
        break;
      }
    }
  }

  /**
   * Test log replication during leader election (Farhan's Implementation)
   */
  async testReplicationDuringElection(): Promise<void> {
    console.log('\n🗳️ REPLICATION DURING ELECTION TEST');
    console.log('-'.repeat(50));

    const leader = await this.findLeader();
    if (!leader) {
      console.log('❌ No leader found');
      return;
    }

    // Execute some commands before election
    console.log('1️⃣ Setting up initial state...');
    await this.executeCommand('set "pre-election" "data"');
    await this.executeCommand('append "pre-election" "-modified"');

    console.log('2️⃣ Getting initial log state...');
    const initialLog = await this.requestLog();
    console.log(`   Initial log entries: ${initialLog?.log.length || 0}`);

    console.log('\n3️⃣ To test election, manually stop the leader:');
    
    // Find which container this leader corresponds to
    const leaderIndex = this.nodes.findIndex(n => n.equals(leader));
    if (leaderIndex >= 0) {
      console.log(`   docker stop raft-node-${leaderIndex + 1}`);
    }

    console.log('\n⏳ Monitoring for new leader...');
    let newLeader: Address | null = null;
    let attempts = 0;
    const maxAttempts = 15;

    while (!newLeader && attempts < maxAttempts) {
      await this.sleep(2000);
      attempts++;
      console.log(`   Attempt ${attempts}/${maxAttempts}...`);
      
      newLeader = await this.findLeader();
      if (newLeader && !newLeader.equals(leader)) {
        console.log(`   🎉 New leader elected: ${newLeader.toString()}`);
        break;
      }
    }

    if (newLeader) {
      console.log('\n4️⃣ Testing log consistency after election...');
      const postElectionLog = await this.requestLog(newLeader);
      
      if (postElectionLog && initialLog) {
        console.log(`   Post-election log entries: ${postElectionLog.log.length}`);
        console.log(`   Log preserved: ${postElectionLog.log.length >= initialLog.log.length ? '✅' : '❌'}`);
      }

      console.log('\n5️⃣ Testing new commands on new leader...');
      const testResult = await this.executeCommand('set "post-election" "test"');
      console.log(`   New command result: ${testResult.success ? '✅' : '❌'} ${testResult.result || testResult.error}`);
    } else {
      console.log('❌ No new leader elected within timeout');
    }
  }

  /**
   * Run assignment demo sequence (Farhan's Implementation)
   */
  async runAssignmentDemo(): Promise<void> {
    console.log('\n🎬 ASSIGNMENT DEMO SEQUENCE');
    console.log('='.repeat(60));

    try {
      // Demo setup
      console.log('\n🏁 Demo Setup');
      await this.getClusterStatus();
      await this.waitForUser();

      // Heartbeat Demo (10 points)
      console.log('\n🫀 HEARTBEAT DEMO (10 points)');
      console.log('1. Testing heartbeat monitoring...');
      await this.testHeartbeat(5);
      
      console.log('\n2. Testing ping connectivity...');
      await this.testPing();
      await this.waitForUser();

      // Log Replication Demo (10 points)  
      console.log('\n📋 LOG REPLICATION DEMO (10 points)');
      await this.testLogReplication();
      await this.waitForUser();

      // Leader Election Demo (10 points)
      console.log('\n🗳️ LEADER ELECTION DEMO (10 points)');
      await this.testLeaderElection();
      await this.waitForUser();

      console.log('\n🏁 Assignment demo completed!');
      console.log('Note: Membership Change demo requires Dhinto\'s implementation');

    } catch (error) {
      console.log(`❌ Demo failed: ${(error as Error).message}`);
    }
  }

  /**
   * Run comprehensive log replication demo (Farhan's Implementation)
   */
  async runLogReplicationDemo(): Promise<void> {
    console.log('\n🎬 COMPREHENSIVE LOG REPLICATION DEMO');
    console.log('='.repeat(60));

    try {
      // 1. Basic log replication test
      await this.testLogReplication();
      await this.waitForUser();

      // 2. Concurrent commands test
      await this.testConcurrentCommands();
      await this.waitForUser();

      // 3. Log consistency test
      await this.testLogConsistency();
      await this.waitForUser();

      // 4. Replication during election test
      await this.testReplicationDuringElection();

      console.log('\n🏁 Log replication demo completed!');

    } catch (error) {
      console.log(`❌ Demo failed: ${(error as Error).message}`);
    }
  }

  /**
   * Run interactive mode with enhanced options
   */
  async runInteractive(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    const question = (prompt: string): Promise<string> => {
      return new Promise((resolve) => {
        rl.question(prompt, resolve);
      });
    };

    while (true) {
      console.log('\n🔧 RAFT TEST CLIENT - Enhanced for Log Replication');
      console.log('-'.repeat(50));
      console.log('1. Cluster Status');
      console.log('2. Find Leader');
      console.log('3. Test Heartbeat');
      console.log('4. Test Ping');
      console.log('5. Test Leader Election');
      console.log('6. Execute Command');
      console.log('7. Request Log');
      console.log('8. Get Log Statistics');
      console.log('9. Test Log Replication');
      console.log('10. Test Concurrent Commands');
      console.log('11. Test Log Consistency');
      console.log('12. Run Assignment Demo');
      console.log('13. Run Full Log Replication Demo');
      console.log('0. Exit');

      const choice = await question('\nSelect option: ');

      switch (choice.trim()) {
        case '1':
          await this.getClusterStatus();
          break;

        case '2':
          const leader = await this.findLeader();
          if (leader) {
            console.log(`✅ Leader found: ${leader.toString()}`);
          }
          break;

        case '3':
          const duration = await question('Duration (seconds, default 10): ');
          const dur = parseInt(duration) || 10;
          await this.testHeartbeat(dur);
          break;

        case '4':
          await this.testPing();
          break;

        case '5':
          await this.testLeaderElection();
          break;

        case '6':
          const cmd = await question('Command: ');
          if (cmd.trim()) {
            const result = await this.executeCommand(cmd.trim());
            console.log(`Result: ${JSON.stringify(result, null, 2)}`);
          }
          break;

        case '7':
          const logResult = await this.requestLog();
          if (logResult) {
            console.log(`📋 Log from leader:`);
            console.log(JSON.stringify(logResult, null, 2));
          }
          break;

        case '8':
          const statsResult = await this.getLogStats();
          if (statsResult) {
            console.log(`📊 Log Statistics:`);
            console.log(JSON.stringify(statsResult, null, 2));
          }
          break;

        case '9':
          await this.testLogReplication();
          break;

        case '10':
          await this.testConcurrentCommands();
          break;

        case '11':
          await this.testLogConsistency();
          break;

        case '12':
          await this.runAssignmentDemo();
          break;

        case '13':
          await this.runLogReplicationDemo();
          break;

        case '0':
          console.log('👋 Goodbye!');
          rl.close();
          return;

        default:
          console.log('❌ Invalid option!');
      }
    }
  }

  /**
   * Demo command execution for assignment compliance
   */
  private async runCommandDemo(): Promise<void> {
    console.log('📝 Testing basic KV commands...');

    const commands = [
      'set key1 value1',
      'get key1',
      'append key1 suffix',
      'get key1',
      'strln key1',
      'set key2 hello',
      'append key2 " world"',
      'get key2',
      'del key1',
      'get key1'
    ];

    for (const cmd of commands) {
      console.log(`\n> ${cmd}`);
      const result = await this.executeCommand(cmd);
      
      if (result.success) {
        console.log(`  ✅ ${result.result}`);
      } else {
        console.log(`  ❌ ${result.error}`);
      }
      
      await this.sleep(1000);
    }
  }

  /**
   * Utility methods
   */
  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async waitForUser(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    return new Promise((resolve) => {
      rl.question('\nPress Enter to continue...', () => {
        rl.close();
        resolve();
      });
    });
  }
}

/**
 * Main CLI entry point with enhanced command support
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const client = new RaftTestClient();

  if (args.length === 0) {
    // Interactive mode
    await client.runInteractive();
  } else {
    // Command mode
    const command = args[0];
    
    switch (command) {
      case 'status':
        await client.getClusterStatus();
        break;
      
      case 'leader':
        const leader = await client.findLeader();
        if (leader) {
          console.log(`Leader: ${leader.toString()}`);
        }
        break;
      
      case 'heartbeat':
        const duration = parseInt(args[1]) || 10;
        await client.testHeartbeat(duration);
        break;
      
      case 'ping':
        await client.testPing();
        break;
      
      case 'election':
        await client.testLeaderElection();
        break;
      
      case 'demo':
        await client.runAssignmentDemo();
        break;
      
      case 'log-replication':
        await client.runLogReplicationDemo();
        break;
      
      case 'execute':
        if (args.length < 2) {
          console.log('❌ Usage: execute <command>');
          process.exit(1);
        }
        const cmd = args.slice(1).join(' ');
        const result = await client.executeCommand(cmd);
        console.log(JSON.stringify(result, null, 2));
        break;
      
      case 'log':
        const logResult = await client.requestLog();
        if (logResult) {
          console.log('📋 Current Log:');
          console.log(`Entries: ${logResult.log.length}`);
          console.log(`Commit Index: ${logResult.metadata.commitIndex}`);
          console.log(`Last Applied: ${logResult.metadata.lastApplied}`);
          console.log(`Current Term: ${logResult.metadata.currentTerm}`);
          
          if (logResult.log.length > 0) {
            console.log('\nRecent entries:');
            logResult.log.slice(-10).forEach((entry: any) => {
              console.log(`[${entry.index}] ${entry.command} ${entry.key || ''}${entry.value ? '=' + entry.value : ''}`);
            });
          }
        }
        break;
      
      case 'stats':
        const statsResult = await client.getLogStats();
        if (statsResult) {
          console.log('📊 Log Statistics:');
          console.log(`Total Entries: ${statsResult.stats.totalEntries}`);
          console.log(`Commit Index: ${statsResult.stats.commitIndex}`);
          console.log(`Command Types: ${JSON.stringify(statsResult.stats.commandTypes)}`);
          console.log(`Term Distribution: ${JSON.stringify(statsResult.stats.termDistribution)}`);
        }
        break;
      
      case 'test-replication':
        await client.testLogReplication();
        break;
      
      case 'test-concurrent':
        await client.testConcurrentCommands();
        break;
      
      case 'test-consistency':
        await client.testLogConsistency();
        break;
      
      case 'help':
        console.log('Available commands:');
        console.log('  status           - Show cluster status');
        console.log('  leader           - Find current leader');
        console.log('  heartbeat [sec]  - Test heartbeat for duration');
        console.log('  ping             - Test ping connectivity');
        console.log('  election         - Test leader election');
        console.log('  demo             - Run assignment demo sequence');
        console.log('  log-replication  - Run comprehensive log replication demo');
        console.log('  execute <cmd>    - Execute a command');
        console.log('  log              - Show current log');
        console.log('  stats            - Show log statistics');
        console.log('  test-replication - Test log replication');
        console.log('  test-concurrent  - Test concurrent commands');
        console.log('  test-consistency - Test log consistency');
        console.log('  help             - Show this help');
        break;
      
      default:
        console.log('❌ Unknown command:', command);
        console.log('Use "help" to see available commands');
        process.exit(1);
    }
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}