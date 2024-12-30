import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"

export default function FileProcessing({ handleFileUpload, isProcessing, progress }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>File Processing</CardTitle>
      </CardHeader>
      <CardContent>
        <Input type="file" onChange={handleFileUpload} accept=".txt,.pdf" />
        {isProcessing && (
          <div className="mt-2">
            <Progress value={progress} className="w-full" />
            <p className="text-center mt-2">{Math.round(progress)}% processed</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

