/**
 * Key-Value Store implementation for Raft consensus protocol
 * Provides in-memory string storage with thread-safe operations
 * FIXED: String parsing and display issues
 */

import { Logger } from '../utils/Logger';

export interface KVCommand {
  type: 'ping' | 'get' | 'set' | 'strln' | 'del' | 'append';
  key?: string;
  value?: string;
  timestamp: number;
}

export interface KVResult {
  success: boolean;
  result?: any;
  error?: string;
}

export class KVStore {
  private data: Map<string, string> = new Map();
  private logger: Logger;
  private readonly mutex = new AsyncMutex();

  constructor(logger?: Logger) {
    this.logger = logger || new Logger({ nodeId: 'kvstore' });
    this.logger.info('KVStore initialized');
  }

  /**
   * Execute a command with proper locking
   */
  async executeCommand(command: KVCommand): Promise<KVResult> {
    return this.mutex.runExclusive(async () => {
      try {
        const result = await this.executeCommandInternal(command);
        
        this.logger.debug('Command executed', {
          command: command.type,
          key: command.key,
          success: result.success
        });

        return result;
      } catch (error) {
        this.logger.error('Command execution failed', error as Error, {
          command: command.type,
          key: command.key
        });

        return {
          success: false,
          error: (error as Error).message
        };
      }
    });
  }

  /**
   * Internal command execution (not thread-safe by itself)
   */
  private async executeCommandInternal(command: KVCommand): Promise<KVResult> {
    switch (command.type) {
      case 'ping':
        return this.ping();
      
      case 'get':
        if (!command.key) {
          throw new Error('GET command requires a key');
        }
        return this.get(command.key);
      
      case 'set':
        if (!command.key || command.value === undefined) {
          throw new Error('SET command requires key and value');
        }
        return this.set(command.key, command.value);
      
      case 'strln':
        if (!command.key) {
          throw new Error('STRLN command requires a key');
        }
        return this.strln(command.key);
      
      case 'del':
        if (!command.key) {
          throw new Error('DEL command requires a key');
        }
        return this.del(command.key);
      
      case 'append':
        if (!command.key || command.value === undefined) {
          throw new Error('APPEND command requires key and value');
        }
        return this.append(command.key, command.value);
      
      default:
        throw new Error(`Unknown command type: ${(command as any).type}`);
    }
  }

  /**
   * Ping - check if store is responsive
   */
  private async ping(): Promise<KVResult> {
    return {
      success: true,
      result: 'PONG'
    };
  }

  /**
   * Get value by key
   */
  private async get(key: string): Promise<KVResult> {
    const value = this.data.get(key) || '';
    
    return {
      success: true,
      result: value
    };
  }

  /**
   * Set key to value
   */
  private async set(key: string, value: string): Promise<KVResult> {
    this.data.set(key, value);
    
    return {
      success: true,
      result: 'OK'
    };
  }

  /**
   * Get string length of value
   */
  private async strln(key: string): Promise<KVResult> {
    const value = this.data.get(key) || '';
    
    return {
      success: true,
      result: value.length
    };
  }

  /**
   * Delete key and return previous value
   */
  private async del(key: string): Promise<KVResult> {
    const previousValue = this.data.get(key) || '';
    this.data.delete(key);
    
    return {
      success: true,
      result: previousValue
    };
  }

  /**
   * Append value to existing key (or create new)
   */
  private async append(key: string, value: string): Promise<KVResult> {
    const currentValue = this.data.get(key) || '';
    const newValue = currentValue + value;
    this.data.set(key, newValue);
    
    return {
      success: true,
      result: 'OK'
    };
  }

  /**
   * Get all keys (for debugging)
   */
  async getKeys(): Promise<string[]> {
    return this.mutex.runExclusive(async () => {
      return Array.from(this.data.keys());
    });
  }

  /**
   * Get store size
   */
  async size(): Promise<number> {
    return this.mutex.runExclusive(async () => {
      return this.data.size;
    });
  }

  /**
   * Clear all data
   */
  async clear(): Promise<void> {
    return this.mutex.runExclusive(async () => {
      this.data.clear();
      this.logger.info('KVStore cleared');
    });
  }

  /**
   * Get complete snapshot (for replication)
   */
  async getSnapshot(): Promise<Record<string, string>> {
    return this.mutex.runExclusive(async () => {
      const snapshot: Record<string, string> = {};
      this.data.forEach((value, key) => {
        snapshot[key] = value;
      });
      return snapshot;
    });
  }

  /**
   * Restore from snapshot (for replication)
   */
  async restoreFromSnapshot(snapshot: Record<string, string>): Promise<void> {
    return this.mutex.runExclusive(async () => {
      this.data.clear();
      Object.entries(snapshot).forEach(([key, value]) => {
        this.data.set(key, value);
      });
      
      this.logger.info('KVStore restored from snapshot', {
        keys: Object.keys(snapshot).length
      });
    });
  }

  /**
   * Parse command from string format (FIXED - Root cause of display issue)
   */
  static parseCommand(commandStr: string): KVCommand {
    const matches = commandStr.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g);
    
    if (!matches || matches.length === 0) {
      throw new Error('Empty command');
    }
    
    // Remove outer quotes if present
    const parts = matches.map(part => {
      if ((part.startsWith('"') && part.endsWith('"')) || 
          (part.startsWith("'") && part.endsWith("'"))) {
        return part.slice(1, -1);
      }
      return part;
    });
    
    const type = parts[0]?.toLowerCase() as KVCommand['type'];
    
    if (!type) {
      throw new Error('Empty command');
    }

    const command: KVCommand = {
      type,
      timestamp: Date.now()
    };

    switch (type) {
      case 'ping':
        break;
      
      case 'get':
      case 'strln':
      case 'del':
        if (parts.length !== 2) {
          throw new Error(`${type.toUpperCase()} command requires exactly 1 argument: key`);
        }
        command.key = parts[1];
        break;
      
      case 'set':
      case 'append':
        if (parts.length !== 3) {
          throw new Error(`${type.toUpperCase()} command requires exactly 2 arguments: key value`);
        }
        command.key = parts[1];
        command.value = parts[2];
        break;
      
      default:
        throw new Error(`Unknown command: ${type}`);
    }

    return command;
  }

  /**
   * Get store statistics
   */
  async getStats(): Promise<{
    totalKeys: number;
    totalValueLength: number;
    averageValueLength: number;
    largestKey: string | null;
    largestValue: string | null;
  }> {
    return this.mutex.runExclusive(async () => {
      let totalValueLength = 0;
      let largestKey = '';
      let largestValue = '';

      this.data.forEach((value, key) => {
        totalValueLength += value.length;
        
        if (key.length > largestKey.length) {
          largestKey = key;
        }
        
        if (value.length > largestValue.length) {
          largestValue = value;
        }
      });

      return {
        totalKeys: this.data.size,
        totalValueLength,
        averageValueLength: this.data.size > 0 ? totalValueLength / this.data.size : 0,
        largestKey: largestKey || null,
        largestValue: largestValue || null
      };
    });
  }
}

/**
 * Simple async mutex implementation for thread safety
 */
class AsyncMutex {
  private mutex = Promise.resolve();

  async runExclusive<T>(callback: () => Promise<T>): Promise<T> {
    const release = await this.acquire();
    try {
      return await callback();
    } finally {
      release();
    }
  }

  private async acquire(): Promise<() => void> {
    let resolve: () => void;
    
    const waitPromise = new Promise<void>((res) => {
      resolve = res;
    });

    const currentMutex = this.mutex;
    this.mutex = currentMutex.then(() => waitPromise);

    await currentMutex;
    
    return resolve!;
  }
}