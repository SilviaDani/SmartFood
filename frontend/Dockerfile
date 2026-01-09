# Frontend Dockerfile - Multi-stage build per distroless

# Stage 1: Builder
FROM node:20-alpine AS builder

WORKDIR /app
COPY . .
RUN npm install --include=dev
RUN npm install -g vite

# Build the Vite app
RUN npm run build

# Stage 2: Runtime (Alpine è leggero)
FROM node:20-alpine

WORKDIR /app

# Copy built application dal builder
COPY --from=builder /app/dist ./dist

# Install un HTTP server leggero
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Serve static files
CMD ["serve", "-s", "dist", "-l", "3000"]