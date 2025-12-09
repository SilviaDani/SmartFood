# Frontend Dockerfile - Multi-stage build per distroless

# Stage 1: Builder
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

# Build the Vite app
COPY frontend/ .
RUN npm run build

# Stage 2: Runtime distroless
FROM gcr.io/distroless/nodejs20-nonroot:nonroot

WORKDIR /app

# Copy built application e node_modules dal builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Expose port
EXPOSE 3000

# Run the app
CMD ["node", "--loader=./node_modules/vite/dist/client/entry.mjs", "dist/index.html"]