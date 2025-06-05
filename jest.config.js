
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/__tests__/**',
    '!src/**/*.test.ts'
  ],
  moduleNameMapping: {
    '^@core/(.*)$': '<rootDir>/src/core/$1',
    '^@app/(.*)$': '<rootDir>/src/app/$1',
    '^@client/(.*)$': '<rootDir>/src/client/$1',
    '^@rpc/(.*)$': '<rootDir>/src/rpc/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1' 
  }
};