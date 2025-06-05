/**
 * Address class for representing node addresses (IP + Port)
 * Used for node identification and communication
 */

export interface AddressData {
  ip: string;
  port: number;
}

export class Address {
  public readonly ip: string;
  public readonly port: number;

  constructor(ip: string, port: number) {
    if (!this.isValidIP(ip)) {
      throw new Error(`Invalid IP address: ${ip}`);
    }
    
    if (!this.isValidPort(port)) {
      throw new Error(`Invalid port: ${port}`);
    }

    this.ip = ip;
    this.port = port;
  }

  /**
   * Create Address from object
   */
  static fromObject(data: AddressData): Address {
    return new Address(data.ip, data.port);
  }

  /**
   * Create Address from string "ip:port"
   */
  static fromString(addressStr: string): Address {
    const parts = addressStr.split(':');
    if (parts.length !== 2) {
      throw new Error(`Invalid address format: ${addressStr}`);
    }
    
    const ip = parts[0]!;
    const port = parseInt(parts[1]!, 10);
    
    return new Address(ip, port);
  }

  /**
   * Get string representation "ip:port"
   */
  toString(): string {
    return `${this.ip}:${this.port}`;
  }

  /**
   * Get object representation
   */
  toObject(): AddressData {
    return {
      ip: this.ip,
      port: this.port
    };
  }

  /**
   * Get URL for HTTP requests
   */
  toURL(protocol: string = 'http'): string {
    return `${protocol}://${this.ip}:${this.port}`;
  }

  /**
   * Check equality with another address
   */
  equals(other: Address | null | undefined): boolean {
    if (!other) return false;
    if (!(other instanceof Address)) return false;
    return this.ip === other.ip && this.port === other.port;
  }

  /**
   * Hash code for use in Sets and Maps
   */
  hashCode(): string {
    return `${this.ip}:${this.port}`;
  }

  /**
   * Validate IP address format
   */
  private isValidIP(ip: string): boolean {
    const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
    return ipRegex.test(ip);
  }

  /**
   * Validate port number
   */
  private isValidPort(port: number): boolean {
    return Number.isInteger(port) && port >= 1 && port <= 65535;
  }

  /**
   * JSON serialization
   */
  toJSON(): AddressData {
    return this.toObject();
  }
}