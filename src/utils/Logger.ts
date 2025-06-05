/**
 * Logging utility for Raft implementation
 * Provides structured logging with node identification
 */

import winston from 'winston';
import { Address } from '../core/Address';

export enum LogLevel {
  ERROR = 'error',
  WARN = 'warn',
  INFO = 'info',
  DEBUG = 'debug'
}

export interface LogContext {
  nodeId?: string;
  address?: Address;
  term?: number;
  nodeType?: string;
  action?: string;
}

export class Logger {
  private logger: winston.Logger;
  private context: LogContext;

  constructor(context: LogContext = {}) {
    this.context = context;
    
    const nodeId = context.nodeId || process.env.NODE_ID || 'unknown';
    const logLevel = process.env.LOG_LEVEL || 'info';
    
    this.logger = winston.createLogger({
      level: logLevel,
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      defaultMeta: {
        nodeId,
        address: context.address?.toString()
      },
      transports: [
        // Console output
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.timestamp({ format: 'HH:mm:ss.SSS' }),
            winston.format.printf(this.formatConsoleMessage.bind(this))
          )
        }),
        
        // File output
        new winston.transports.File({
          filename: `/app/logs/raft-node-${nodeId}.log`,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          )
        }),
        
        // Error file
        new winston.transports.File({
          filename: `/app/logs/raft-node-${nodeId}-error.log`,
          level: 'error',
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          )
        })
      ]
    });
  }

  /**
   * Update logging context
   */
  updateContext(context: Partial<LogContext>): void {
    this.context = { ...this.context, ...context };
  }

  /**
   * Log info message
   */
  info(message: string, meta: object = {}): void {
    this.logger.info(message, { ...this.context, ...meta });
  }

  /**
   * Log error message
   */
  error(message: string, error?: Error, meta: object = {}): void {
    this.logger.error(message, { 
      ...this.context, 
      ...meta, 
      error: error?.message,
      stack: error?.stack
    });
  }

  /**
   * Log warning message
   */
  warn(message: string, meta: object = {}): void {
    this.logger.warn(message, { ...this.context, ...meta });
  }

  /**
   * Log debug message
   */
  debug(message: string, meta: object = {}): void {
    this.logger.debug(message, { ...this.context, ...meta });
  }

  /**
   * Log Raft-specific events
   */
  raftEvent(event: string, details: object = {}): void {
    this.info(`[RAFT] ${event}`, { 
      event, 
      ...details,
      timestamp: Date.now()
    });
  }

  /**
   * Log election events
   */
  election(message: string, details: object = {}): void {
    this.info(`[ELECTION] ${message}`, { 
      ...details,
      category: 'election'
    });
  }

  /**
   * Log heartbeat events
   */
  heartbeat(message: string, details: object = {}): void {
    this.debug(`[HEARTBEAT] ${message}`, { 
      ...details,
      category: 'heartbeat'
    });
  }

  /**
   * Log membership events
   */
  membership(message: string, details: object = {}): void {
    this.info(`[MEMBERSHIP] ${message}`, { 
      ...details,
      category: 'membership'
    });
  }

  /**
   * Log log replication events
   */
  logReplication(message: string, details: object = {}): void {
    this.info(`[LOG_REPLICATION] ${message}`, { 
      ...details,
      category: 'log_replication'
    });
  }

  /**
   * Format console message for better readability
   */
  private formatConsoleMessage(info: winston.Logform.TransformableInfo): string {
    const timestamp = info.timestamp as string;
    const level = info.level;
    const message = info.message as string;
    const nodeId = info.nodeId as string;
    const address = info.address as string;
    const nodeType = info.nodeType as string;
    
    let prefix = `[${timestamp}]`;
    
    if (address) {
      prefix += ` [${address}]`;
    } else if (nodeId) {
      prefix += ` [${nodeId}]`;
    }
    
    if (nodeType) {
      prefix += ` [${nodeType}]`;
    }
    
    return `${prefix} ${level.toUpperCase()}: ${message}`;
  }

  /**
   * Create child logger with additional context
   */
  child(additionalContext: LogContext): Logger {
    return new Logger({ ...this.context, ...additionalContext });
  }

  /**
   * Get the underlying winston logger instance
   */
  getWinstonLogger(): winston.Logger {
    return this.logger;
  }
}