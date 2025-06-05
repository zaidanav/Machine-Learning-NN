/**
 * Enhanced HTTP-based RPC client for Raft communication with Log Replication support
 * Enhanced by Farhan for better log replication handling
 */

// Import fetch for Node.js
const fetch = require('node-fetch');

export interface RpcRequest {
  jsonrpc: '2.0';
  method: string;
  params: any;
  id: string | number;
}

export interface RpcResponse {
  jsonrpc: '2.0';
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string | number;
}

export interface RpcCallOptions {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

export class SimpleRpcClient {
  private timeout: number;
  private defaultRetries: number;
  private defaultRetryDelay: number;

  constructor(timeout: number = 1000, retries: number = 2, retryDelay: number = 100) {
    this.timeout = timeout;
    this.defaultRetries = retries;
    this.defaultRetryDelay = retryDelay;
  }

  /**
   * Make RPC call to target URL with retry logic
   */
  async call(url: string, method: string, params: any, options?: RpcCallOptions): Promise<any> {
    const opts = {
      timeout: options?.timeout || this.timeout,
      retries: options?.retries || this.defaultRetries,
      retryDelay: options?.retryDelay || this.defaultRetryDelay
    };

    let lastError: Error;

    for (let attempt = 0; attempt <= opts.retries; attempt++) {
      try {
        return await this.executeCall(url, method, params, opts.timeout);
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on certain errors
        if (this.shouldNotRetry(error as Error)) {
          throw error;
        }

        // Don't retry on the last attempt
        if (attempt === opts.retries) {
          break;
        }

        // Wait before retrying
        if (opts.retryDelay > 0) {
          await this.sleep(opts.retryDelay * (attempt + 1)); // Exponential backoff
        }
      }
    }

    throw lastError!;
  }

  /**
   * Execute single RPC call
   */
  private async executeCall(url: string, method: string, params: any, timeout: number): Promise<any> {
    const requestId = Math.floor(Math.random() * 1000000);
    
    const request: RpcRequest = {
      jsonrpc: '2.0',
      method,
      params: params || {},
      id: requestId
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${url}/rpc`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'User-Agent': 'RaftNode-RPC-Client/1.0'
        },
        body: JSON.stringify(request),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseText = await response.text();
      let rpcResponse: RpcResponse;
      
      try {
        rpcResponse = JSON.parse(responseText);
      } catch (parseError) {
        throw new Error(`Invalid JSON response: ${responseText.substring(0, 100)}`);
      }

      // Validate RPC response format
      if (!rpcResponse.jsonrpc || rpcResponse.id !== requestId) {
        throw new Error('Invalid RPC response format');
      }

      if (rpcResponse.error) {
        const error = new Error(`RPC Error ${rpcResponse.error.code}: ${rpcResponse.error.message}`);
        (error as any).code = rpcResponse.error.code;
        (error as any).data = rpcResponse.error.data;
        throw error;
      }

      return rpcResponse.result;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`RPC timeout after ${timeout}ms`);
      }
      throw error;
    }
  }

  /**
   * Determine if error should not be retried
   */
  private shouldNotRetry(error: Error): boolean {
    // Don't retry on certain HTTP status codes
    if (error.message.includes('HTTP 400') || 
        error.message.includes('HTTP 401') || 
        error.message.includes('HTTP 403') ||
        error.message.includes('HTTP 404')) {
      return true;
    }

    // Don't retry on certain RPC errors
    if (error.message.includes('RPC Error -32601') || // Method not found
        error.message.includes('RPC Error -32600')) { // Invalid request
      return true;
    }

    return false;
  }

  /**
   * Test connection to target
   */
  async ping(url: string): Promise<boolean> {
    try {
      await this.call(url, 'ping', {}, { timeout: 2000, retries: 1 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Batch RPC calls for efficiency
   */
  async batchCall(url: string, calls: Array<{ method: string; params: any }>): Promise<any[]> {
    const batchRequest = calls.map((call, index) => ({
      jsonrpc: '2.0' as const,
      method: call.method,
      params: call.params || {},
      id: index
    }));

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${url}/rpc`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(batchRequest),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const batchResponse = await response.json();
      
      if (!Array.isArray(batchResponse)) {
        throw new Error('Invalid batch response format');
      }

      return batchResponse.map((resp: RpcResponse) => {
        if (resp.error) {
          throw new Error(`RPC Error ${resp.error.code}: ${resp.error.message}`);
        }
        return resp.result;
      });

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Batch RPC timeout after ${this.timeout}ms`);
      }
      throw error;
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get client configuration
   */
  getConfig(): { timeout: number; retries: number; retryDelay: number } {
    return {
      timeout: this.timeout,
      retries: this.defaultRetries,
      retryDelay: this.defaultRetryDelay
    };
  }

  /**
   * Update client configuration
   */
  updateConfig(config: Partial<{ timeout: number; retries: number; retryDelay: number }>): void {
    if (config.timeout !== undefined) {
      this.timeout = config.timeout;
    }
    if (config.retries !== undefined) {
      this.defaultRetries = config.retries;
    }
    if (config.retryDelay !== undefined) {
      this.defaultRetryDelay = config.retryDelay;
    }
  }
}