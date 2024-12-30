import React from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Download, Upload } from 'lucide-react'

export default function KnowledgeManagement({ handleSave, handleLoad }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Knowledge Management</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex space-x-2">
          <Button onClick={handleSave}>
            <Download className="mr-2 h-4 w-4" />
            Save Knowledge
          </Button>
          <Input type="file" accept=".json" onChange={handleLoad} className="hidden" id="load-knowledge" />
          <Button onClick={() => document.getElementById('load-knowledge')?.click()}>
            <Upload className="mr-2 h-4 w-4" />
            Load Knowledge
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

