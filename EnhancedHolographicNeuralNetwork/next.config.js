/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://integrate.api.nvidia.com/v1/:path*',
      },
    ]
  },
}

module.exports = nextConfig

