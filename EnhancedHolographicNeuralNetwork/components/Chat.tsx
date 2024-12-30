import React from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function Chat({ chatHistory, inputText, setInputText, handleSubmit, predictedWords }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Enhanced Holographic Neural Network Chat</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4 mb-4">
          {chatHistory.map((message, index) => (
            <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-xs p-2 rounded-lg ${message.type === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
                {message.text}
              </div>
            </div>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <Input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your message..."
            className="flex-grow"
          />
          <Button type="submit">Send</Button>
        </form>
        {predictedWords.length > 0 && (
          <div className="mt-2 text-sm text-gray-500">
            Predicted words: {predictedWords.join(', ')}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

