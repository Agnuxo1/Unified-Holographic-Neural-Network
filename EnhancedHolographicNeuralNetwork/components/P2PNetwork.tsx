import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

export default function P2PNetwork({ nodeId, p2pConnected, connectedPeers }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>P2P Network</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <Input
            type="text"
            value={nodeId}
            readOnly
            placeholder="Your Node ID"
          />
          <div>{p2pConnected ? 'Connected to P2P network' : 'Disconnected from P2P network'}</div>
          <div>Connected Peers: {connectedPeers.length}</div>
        </div>
      </CardContent>
    </Card>
  )
}

