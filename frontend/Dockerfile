FROM node:18-alpine
WORKDIR /app
COPY package.json ./
RUN npm install
COPY src/ ./src/
EXPOSE 3000
CMD ["npx", "http-server", "src", "-p", "3000", "--cors"]
