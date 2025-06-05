import express from 'express';
import cors from 'cors';
import { Address } from '../core/Address';
import { 
  RaftNode, 
  VoteRequest, 
  HeartbeatRequest, 
  LogReplicationRequest, 
  MembershipRequest 
} from '../core/RaftNode';
import { KVStore } from './KVStore';
import { Logger } from '../utils/Logger';
import { SimpleRpcClient } from '../rpc/SimpleRpcClient';

// Add fetch polyfill for Node.js
const fetch = require('node-fetch');
if (!globalThis.fetch) {
  globalThis.fetch = fetch;
}

interface ServerConfig {
  nodeIp: string;
  nodePort: number;
  contactIp?: string;
  contactPort?: number;
  nodeId: string;
  clusterInit?: boolean;
}

interface RpcRequestBody {
  method: string;
  params: unknown;
  id: string | number;
  jsonrpc?: string;
}

interface ExecuteRequestBody {
  command: string;
}

interface LogStats {
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
}

interface MembershipAddRequest {
  nodeAddress: {
    ip: string;
    port: number;
  };
}

interface MembershipRemoveRequest {
  nodeAddress: {
    ip: string;
    port: number;
  };
}

export class RaftServer {
  private app: express.Application;
  private raftNode: RaftNode;
  private kvStore: KVStore;
  private logger: Logger;
  private config: ServerConfig;
  private server: any;

  constructor(config: ServerConfig) {
    this.config = config;
    
    // Initialize logger
    this.logger = new Logger({
      nodeId: config.nodeId,
      address: new Address(config.nodeIp, config.nodePort)
    });

    // Initialize KV Store
    this.kvStore = new KVStore(this.logger);

    // Initialize RPC client
    const rpcClient = new SimpleRpcClient();

    // Initialize Raft Node with KVStore (Farhan's integration)
    const nodeAddress = new Address(config.nodeIp, config.nodePort);
    this.raftNode = new RaftNode(
      nodeAddress,
      this.kvStore,  // Pass KVStore to RaftNode for log replication
      this.logger,
      rpcClient
    );

    // Initialize Express app
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();

    this.logger.info('Raft server initialized with log replication and membership change', {
      nodeId: config.nodeId,
      address: nodeAddress.toString(),
      clusterInit: config.clusterInit
    });
  }

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    this.app.use(cors());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Request logging
    this.app.use((req, res, next) => {
      this.logger.debug('HTTP Request', {
        method: req.method,
        path: req.path,
        ip: req.ip
      });
      next();
    });
  }

  /**
   * Validate and cast RPC parameters
   */
  private validateRpcParams<T>(params: unknown, method: string): T {
    if (!params || typeof params !== 'object') {
      throw new Error(`Invalid parameters for method ${method}`);
    }
    return params as T;
  }

  /**
   * Setup Express routes with enhanced log replication support and FIXED membership endpoints
   */
  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        nodeId: this.config.nodeId,
        address: `${this.config.nodeIp}:${this.config.nodePort}`,
        raftStatus: this.raftNode.getClusterInfo(),
        timestamp: new Date().toISOString()
      });
    });

    // RPC endpoint for inter-node communication (Enhanced with membership RPCs)
    this.app.post('/rpc', async (req, res) => {
      try {
        const body = req.body as RpcRequestBody;
        const { method, params, id } = body;
        
        // Validate RPC request format
        if (!method || id === undefined || id === null) {
          res.status(400).json({
            jsonrpc: '2.0',
            error: {
              code: -32600,
              message: 'Invalid Request - missing method or id'
            },
            id: id || null
          });
          return;
        }

        let result: unknown;

        this.logger.debug(`Received RPC call: ${method}`, { params });

        switch (method) {
          case 'requestVote':
            try {
              const voteRequest = this.validateRpcParams<VoteRequest>(params, method);
              result = await this.raftNode.handleVoteRequest(voteRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for requestVote: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;
          
          case 'heartbeat':
            try {
              const heartbeatRequest = this.validateRpcParams<HeartbeatRequest>(params, method);
              result = await this.raftNode.handleHeartbeat(heartbeatRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for heartbeat: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;
          
          case 'logReplication':  // Farhan's log replication handler
            try {
              const logReplicationRequest = this.validateRpcParams<LogReplicationRequest>(params, method);
              result = await this.raftNode.handleLogReplication(logReplicationRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for logReplication: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;
          
          case 'applyMembership':
            try {
              const membershipRequest = this.validateRpcParams<MembershipRequest>(params, method);
              result = await this.raftNode.handleMembershipApplication(membershipRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for applyMembership: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;
          
          case 'syncClusterState':  // NEW - For membership synchronization
            try {
              const clusterStateRequest = this.validateRpcParams<any>(params, method);
              result = await this.raftNode.handleClusterStateSync(clusterStateRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for syncClusterState: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;

          case 'membershipUpdate':  // Add this case in Server.ts RPC handler
            try {
              const membershipUpdateRequest = this.validateRpcParams<any>(params, method);
              result = await this.raftNode.handleMembershipUpdate(membershipUpdateRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for membershipUpdate: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;

          case 'notifyRemoval':  // Add this case too
            try {
              const removalRequest = this.validateRpcParams<any>(params, method);
              result = await this.raftNode.handleRemovalNotification(removalRequest);
            } catch (validationError) {
              res.status(400).json({
                jsonrpc: '2.0',
                error: {
                  code: -32602,
                  message: `Invalid params for notifyRemoval: ${(validationError as Error).message}`
                },
                id
              });
              return;
            }
            break;
          
          case 'ping':
            result = { pong: true, timestamp: Date.now() };
            break;
          
          default:
            res.status(400).json({
              jsonrpc: '2.0',
              error: {
                code: -32601,
                message: `Unknown RPC method: ${method}`
              },
              id
            });
            return;
        }

        res.json({
          jsonrpc: '2.0',
          result,
          id
        });

        return;

      } catch (error) {
        this.logger.error('RPC request failed', error as Error);
        res.json({
          jsonrpc: '2.0',
          error: {
            code: -32603,
            message: (error as Error).message
          },
          id: (req.body as RpcRequestBody)?.id || null
        });
        return;
      }
    });

    // ===============================
    // MEMBERSHIP CHANGE ENDPOINTS (NEW - FIXED)
    // ===============================

    // Manual membership change - ADD NODE
    this.app.post('/membership/add', async (req, res) => {
      try {
        if (!this.raftNode.isLeader()) {
          const leader = this.raftNode.getLeaderAddress();
          res.status(403).json({
            success: false,
            error: 'Not the leader',
            leader: leader?.toString(),
            redirect: leader ? `http://${leader.toString()}/membership/add` : null,
            message: 'Only the leader can add nodes to the cluster'
          });
          return;
        }

        const { nodeAddress } = req.body as MembershipAddRequest;
        if (!nodeAddress || !nodeAddress.ip || !nodeAddress.port) {
          res.status(400).json({
            success: false,
            error: 'Invalid node address format',
            message: 'nodeAddress must contain ip and port fields'
          });
          return;
        }

        this.logger.membership('Processing add node request', {
          targetNode: `${nodeAddress.ip}:${nodeAddress.port}`,
          requestedBy: req.ip
        });

        const targetAddress = new Address(nodeAddress.ip, nodeAddress.port);
        const success = await this.raftNode.addNodeToCluster(targetAddress);

        const currentClusterAddresses = this.raftNode.getClusterAddresses();

        res.json({
          success,
          message: success ? 'Node added successfully' : 'Failed to add node',
          clusterSize: currentClusterAddresses.length,
          clusterMembers: currentClusterAddresses.map(a => a.toObject()),
          addedNode: success ? nodeAddress : null,
          timestamp: Date.now()
        });

      } catch (error) {
        this.logger.error('Add membership failed', error as Error);
        res.status(500).json({
          success: false,
          error: (error as Error).message,
          message: 'Internal error during node addition'
        });
      }
    });

    // Manual membership change - REMOVE NODE
    this.app.post('/membership/remove', async (req, res) => {
      // 1. Leadership check
      if (!this.raftNode.isLeader()) {
        const leader = this.raftNode.getLeaderAddress();
        res.status(403).json({
          success: false,
          error: 'Not the leader',
          leader: leader?.toString()
        });
        return;
      }

      const { nodeAddress } = req.body;

      // 2. Request validation
      if (!nodeAddress || !nodeAddress.ip || !nodeAddress.port) {
        res.status(400).json({
          success: false,
          error: 'Invalid node address format',
          message: 'nodeAddress must contain ip and port fields'
        });
        return;
      }

      // 3. IP validation
      const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
      if (!ipRegex.test(nodeAddress.ip)) {
        res.status(400).json({
          success: false,
          error: 'Invalid IP address format'
        });
        return;
      }

      // 4. Port validation
      if (!Number.isInteger(nodeAddress.port) || nodeAddress.port < 1 || nodeAddress.port > 65535) {
        res.status(400).json({
          success: false,
          error: 'Invalid port number'
        });
        return;
      }

      // 5. Execute removal
      const targetAddress = new Address(nodeAddress.ip, nodeAddress.port);
      const success = await this.raftNode.removeNodeFromCluster(targetAddress);
      
      res.json({
        success,
        message: success ? 'Node removed successfully' : 'Failed to remove node',
        clusterSize: this.raftNode.getClusterAddresses().length,
        removedNode: success ? nodeAddress : null
      });
    });

    // Get cluster membership information
    this.app.get('/membership/info', (req, res) => {
      try {
        const clusterInfo = this.raftNode.getClusterInfo();
        const clusterAddresses = this.raftNode.getClusterAddresses();
        const selfAddress = new Address(this.config.nodeIp, this.config.nodePort);

        res.json({
          success: true,
          clusterInfo: {
            size: clusterAddresses.length,
            members: clusterAddresses.map(addr => addr.toObject()),
            leader: clusterInfo.leader,
            self: clusterInfo.address,
            term: clusterInfo.currentTerm,
            type: clusterInfo.type,
            commitIndex: clusterInfo.commitIndex,
            lastLogIndex: clusterInfo.lastLogIndex
          },
          memberDetails: clusterAddresses.map(addr => ({
            address: addr.toString(),
            isLeader: addr.equals(this.raftNode.getLeaderAddress()),
            isSelf: addr.equals(selfAddress),
            status: 'active' // TODO: Implement proper status tracking
          })),
          timestamp: Date.now()
        });

      } catch (error) {
        this.logger.error('Get membership info failed', error as Error);
        res.status(500).json({
          success: false,
          error: (error as Error).message,
          message: 'Failed to retrieve membership information'
        });
      }
    });

    // ===============================
    // CLIENT INTERFACE ENDPOINTS
    // ===============================

    // Client interface - execute commands (Enhanced by Farhan for Log Replication)
    this.app.post('/execute', async (req, res) => {
      try {
        // Check if this node is the leader
        if (!this.raftNode.isLeader()) {
          const leader = this.raftNode.getLeaderAddress();
          
          this.logger.warn('Command received by non-leader node', {
            command: (req.body as ExecuteRequestBody)?.command,
            actualLeader: leader?.toString()
          });
          
          res.status(403).json({
            success: false,
            error: 'Not the leader',
            leader: leader?.toString(),
            redirect: leader ? `http://${leader.toString()}/execute` : null,
            message: 'Please redirect your request to the current leader'
          });
          return;
        }

        const { command } = req.body as ExecuteRequestBody;
        
        if (!command || typeof command !== 'string') {
          res.status(400).json({
            success: false,
            error: 'Invalid command format',
            message: 'Command must be a non-empty string'
          });
          return;
        }

        this.logger.info('Executing client command', { 
          command: command.substring(0, 100), // Truncate for logging
          clientIp: req.ip 
        });

        try {
          // Execute command through Raft consensus (Farhan's implementation)
          const result = await this.raftNode.execute(command);

          this.logger.info('Command executed successfully', {
            command: command.split(' ')[0], // Log only command type
            success: true
          });

          res.json({
            success: true,
            result: result.result || result,
            timestamp: Date.now(),
            leader: this.raftNode.getClusterInfo().address
          });
          return;

        } catch (executeError) {
          const error = executeError as Error;
          
          // Check if error is about leadership
          if (error.message.includes('Not the leader')) {
            const leader = this.raftNode.getLeaderAddress();
            
            res.status(503).json({
              success: false,
              error: 'Leadership changed during execution',
              leader: leader?.toString(),
              redirect: leader ? `http://${leader.toString()}/execute` : null,
              message: 'Please retry with the current leader'
            });
            return;
          }

          this.logger.error('Command execution failed', error, {
            command: command.split(' ')[0]
          });

          res.status(500).json({
            success: false,
            error: error.message,
            timestamp: Date.now()
          });
          return;
        }

      } catch (error) {
        this.logger.error('Execute endpoint error', error as Error);
        res.status(500).json({
          success: false,
          error: 'Internal server error',
          message: (error as Error).message
        });
        return;
      }
    });

    // Get current log (Enhanced by Farhan for detailed log information)
    this.app.get('/request_log', (req, res) => {
      try {
        // Check if this node is the leader
        if (!this.raftNode.isLeader()) {
          const leader = this.raftNode.getLeaderAddress();
          
          this.logger.warn('Log request received by non-leader node', {
            actualLeader: leader?.toString()
          });
          
          res.status(403).json({
            success: false,
            error: 'Not the leader',
            leader: leader?.toString(),
            redirect: leader ? `http://${leader.toString()}/request_log` : null,
            message: 'Log requests must be sent to the leader'
          });
          return;
        }

        // Get detailed log information (Farhan's implementation)
        const detailedLog = this.raftNode.getDetailedLog();
        const clusterInfo = this.raftNode.getClusterInfo();

        this.logger.info('Serving log request', {
          entriesCount: detailedLog.entries.length,
          commitIndex: detailedLog.commitIndex,
          requestIp: req.ip
        });

        res.json({
          success: true,
          log: detailedLog.entries,
          metadata: {
            commitIndex: detailedLog.commitIndex,
            lastApplied: detailedLog.lastApplied,
            logLength: detailedLog.logLength,
            currentTerm: clusterInfo.currentTerm,
            clusterSize: clusterInfo.clusterSize
          },
          clusterInfo,
          timestamp: Date.now(),
          leader: clusterInfo.address
        });
        return;

      } catch (error) {
        this.logger.error('Request log failed', error as Error);
        res.status(500).json({
          success: false,
          error: 'Failed to retrieve log',
          message: (error as Error).message
        });
        return;
      }
    });

    // Get cluster information
    this.app.get('/cluster_info', (req, res) => {
      try {
        const info = this.raftNode.getClusterInfo();
        res.json(info);
      } catch (error) {
        this.logger.error('Get cluster info failed', error as Error);
        res.status(500).json({
          error: (error as Error).message
        });
      }
    });

    // Log statistics endpoint (Farhan's implementation)
    this.app.get('/log_stats', (req, res) => {
      try {
        if (!this.raftNode.isLeader()) {
          const leader = this.raftNode.getLeaderAddress();
          res.status(403).json({
            error: 'Not the leader',
            leader: leader?.toString(),
            redirect: leader ? `http://${leader.toString()}/log_stats` : null
          });
          return;
        }

        const log = this.raftNode.getLog();
        const clusterInfo = this.raftNode.getClusterInfo();

        const stats: LogStats = {
          totalEntries: log.length,
          commitIndex: clusterInfo.commitIndex,
          lastLogIndex: clusterInfo.lastLogIndex,
          termDistribution: log.reduce((acc, entry) => {
            acc[entry.term] = (acc[entry.term] || 0) + 1;
            return acc;
          }, {} as Record<number, number>),
          commandTypes: log.reduce((acc, entry) => {
            acc[entry.command] = (acc[entry.command] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
          recentEntries: log.slice(-10).map(entry => ({
            index: entry.index,
            term: entry.term,
            command: entry.command,
            timestamp: new Date(entry.timestamp).toISOString()
          }))
        };

        res.json({
          success: true,
          stats,
          timestamp: Date.now()
        });

      } catch (error) {
        this.logger.error('Log stats failed', error as Error);
        res.status(500).json({
          success: false,
          error: (error as Error).message
        });
      }
    });

    // KV Store statistics (for debugging)
    this.app.get('/kvstore/stats', async (req, res) => {
      try {
        const stats = await this.kvStore.getStats();
        const snapshot = await this.kvStore.getSnapshot();
        
        res.json({
          stats,
          snapshot,
          timestamp: Date.now()
        });
      } catch (error) {
        this.logger.error('Get KV stats failed', error as Error);
        res.status(500).json({
          error: (error as Error).message
        });
      }
    });

    // Debug endpoint for replication status (Farhan's implementation)
    this.app.get('/replication_status', (req, res) => {
      try {
        if (!this.raftNode.isLeader()) {
          const leader = this.raftNode.getLeaderAddress();
          res.status(403).json({
            error: 'Not the leader',
            leader: leader?.toString()
          });
          return;
        }

        const clusterInfo = this.raftNode.getClusterInfo();
        const log = this.raftNode.getLog();
        
        // Basic replication status
        const replicationStatus = {
          isLeader: true,
          currentTerm: clusterInfo.currentTerm,
          logLength: log.length,
          commitIndex: clusterInfo.commitIndex,
          lastLogIndex: clusterInfo.lastLogIndex,
          clusterSize: clusterInfo.clusterSize,
          timestamp: Date.now()
        };

        res.json({
          success: true,
          replicationStatus
        });

      } catch (error) {
        this.logger.error('Get replication status failed', error as Error);
        res.status(500).json({
          success: false,
          error: (error as Error).message
        });
      }
    });

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        error: 'Endpoint not found',
        path: req.originalUrl,
        availableEndpoints: [
          'GET /health',
          'POST /rpc',
          'POST /execute',
          'GET /request_log',
          'GET /cluster_info',
          'GET /log_stats',
          'GET /membership/info',
          'POST /membership/add',
          'POST /membership/remove',
          'GET /replication_status',
          'GET /kvstore/stats'
        ]
      });
    });
  }

  /**
   * Setup error handling
   */
  private setupErrorHandling(): void {
    this.app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
      this.logger.error('Unhandled HTTP error', error);
      res.status(500).json({
        error: 'Internal server error',
        message: error.message
      });
    });

    // Graceful shutdown handlers
    process.on('SIGTERM', () => this.shutdown('SIGTERM'));
    process.on('SIGINT', () => this.shutdown('SIGINT'));
    process.on('uncaughtException', (error) => {
      this.logger.error('Uncaught exception', error);
      this.shutdown('uncaughtException');
    });
    process.on('unhandledRejection', (reason) => {
      this.logger.error('Unhandled rejection', reason as Error);
    });
  }

  /**
   * Start the server
   */
  async start(): Promise<void> {
    try {
      // Start HTTP server
      this.server = this.app.listen(this.config.nodePort, this.config.nodeIp, () => {
        this.logger.info('HTTP server started', {
          address: `${this.config.nodeIp}:${this.config.nodePort}`
        });
      });

      // Start Raft node
      const contactAddress = this.config.contactIp && this.config.contactPort
        ? new Address(this.config.contactIp, this.config.contactPort)
        : undefined;

      await this.raftNode.start(contactAddress);

      this.logger.info('Raft server started successfully', {
        nodeId: this.config.nodeId,
        isLeader: this.raftNode.isLeader(),
        clusterSize: this.raftNode.getClusterAddresses().length
      });

      this.printStartupBanner();

    } catch (error) {
      this.logger.error('Failed to start server', error as Error);
      throw error;
    }
  }

  /**
   * Shutdown the server gracefully
   */
  async shutdown(signal: string): Promise<void> {
    this.logger.info('Shutting down server', { signal });

    try {
      // Stop Raft node
      await this.raftNode.stop();

      // Close HTTP server
      if (this.server) {
        await new Promise<void>((resolve) => {
          this.server.close(() => {
            this.logger.info('HTTP server closed');
            resolve();
          });
        });
      }

      this.logger.info('Server shutdown complete');
      process.exit(0);

    } catch (error) {
      this.logger.error('Error during shutdown', error as Error);
      process.exit(1);
    }
  }

  /**
   * Print startup banner
   */
  private printStartupBanner(): void {
    const clusterInfo = this.raftNode.getClusterInfo();
    
    console.log('\n' + '='.repeat(60));
    console.log('          RAFT CONSENSUS PROTOCOL SERVER');
    console.log('          Enhanced with Log Replication & Membership');
    console.log('='.repeat(60));
    console.log(`🚀 Node ID: ${this.config.nodeId}`);
    console.log(`📍 Address: ${this.config.nodeIp}:${this.config.nodePort}`);
    console.log(`👑 Role: ${clusterInfo.type}`);
    console.log(`📊 Term: ${clusterInfo.currentTerm}`);
    console.log(`🔗 Cluster Size: ${clusterInfo.clusterSize}`);
    console.log(`📋 Log Entries: ${clusterInfo.lastLogIndex + 1}`);
    console.log(`✅ Commit Index: ${clusterInfo.commitIndex}`);
    
    if (clusterInfo.leader) {
      console.log(`🎯 Leader: ${clusterInfo.leader.ip}:${clusterInfo.leader.port}`);
    }
    
    console.log(`🌐 HTTP: http://${this.config.nodeIp}:${this.config.nodePort}`);
    console.log(`📋 Health: http://${this.config.nodeIp}:${this.config.nodePort}/health`);
    console.log(`📊 Log Stats: http://${this.config.nodeIp}:${this.config.nodePort}/log_stats`);
    console.log(`👥 Membership: http://${this.config.nodeIp}:${this.config.nodePort}/membership/info`);
    console.log('='.repeat(60));
    console.log('💡 Use Ctrl+C to stop gracefully');
    console.log('🔧 Membership endpoints: /membership/add, /membership/remove');
    console.log('-'.repeat(60) + '\n');
  }
}

/**
 * Parse command line arguments and environment variables
 */
function parseConfig(): ServerConfig {
  const args = process.argv.slice(2);
  
  // Parse from environment variables (Docker mode)
  if (process.env.NODE_IP && process.env.NODE_PORT) {
    return {
      nodeIp: process.env.NODE_IP,
      nodePort: parseInt(process.env.NODE_PORT, 10),
      contactIp: process.env.CONTACT_IP,
      contactPort: process.env.CONTACT_PORT ? parseInt(process.env.CONTACT_PORT, 10) : undefined,
      nodeId: process.env.NODE_ID || 'unknown',
      clusterInit: process.env.CLUSTER_INIT === 'true'
    };
  }

  // Parse from command line arguments
  if (args.length < 2) {
    console.error('Usage: npm start <ip> <port> [contact_ip] [contact_port]');
    console.error('Example:');
    console.error('  npm start 172.20.0.10 8001                    # First node');
    console.error('  npm start 172.20.0.11 8001 172.20.0.10 8001  # Join cluster');
    process.exit(1);
  }

  const nodeIp = args[0]!;
  const nodePort = parseInt(args[1]!, 10);
  const contactIp = args[2];
  const contactPort = args[3] ? parseInt(args[3], 10) : undefined;

  return {
    nodeIp,
    nodePort,
    contactIp,
    contactPort,
    nodeId: `${nodeIp}:${nodePort}`,
    clusterInit: !contactIp
  };
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  try {
    const config = parseConfig();
    const server = new RaftServer(config);
    await server.start();
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start server if this file is run directly
if (require.main === module) {
  main().catch(console.error);
}