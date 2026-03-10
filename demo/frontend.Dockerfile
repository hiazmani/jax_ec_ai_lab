FROM node:20-slim

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

EXPOSE 5173

# Vite needs --host to be accessible from outside the container
CMD ["npm", "run", "dev", "--", "--host"]
