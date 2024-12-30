import React from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export default function Training({ isTraining, trainingProgress, handleTrain }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Training</CardTitle>
      </CardHeader>
      <CardContent>
        <Button onClick={handleTrain} disabled={isTraining}>
          {isTraining ? 'Training...' : 'Train'}
        </Button>
        {isTraining && (
          <div className="mt-2">
            <Progress value={trainingProgress} className="w-full" />
            <p className="text-center mt-2">{trainingProgress}% completed</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

