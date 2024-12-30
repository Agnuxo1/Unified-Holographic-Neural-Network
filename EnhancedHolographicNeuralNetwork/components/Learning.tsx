import React from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function Learning({ learnInput, setLearnInput, learnResponse, setLearnResponse, handleLearn }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Learning</CardTitle>
      </CardHeader>
      <CardContent>
        <Input
          type="text"
          value={learnInput}
          onChange={(e) => setLearnInput(e.target.value)}
          placeholder="Input to learn"
          className="mb-2"
        />
        <Input
          type="text"
          value={learnResponse}
          onChange={(e) => setLearnResponse(e.target.value)}
          placeholder="Associated response"
          className="mb-2"
        />
        <Button onClick={handleLearn}>Learn</Button>
      </CardContent>
    </Card>
  )
}

