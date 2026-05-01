/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Docker 多阶段构建需要 standalone 产物（见 frontend/Dockerfile）
  output: 'standalone',
}

module.exports = nextConfig
