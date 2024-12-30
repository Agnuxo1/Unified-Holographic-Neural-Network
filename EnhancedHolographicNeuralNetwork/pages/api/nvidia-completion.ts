import { NextApiRequest, NextApiResponse } from 'next'
import { OpenAI } from 'openai'

const API_KEY = process.env.NVIDIA_API_KEY

const nvidiaClient = new OpenAI({
  baseURL: "https://integrate.api.nvidia.com/v1",
  apiKey: API_KEY
})

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { prompt } = req.body

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' })
  }

  try {
    const completion = await nvidiaClient.chat.completions.create({
      model: "nvidia/llama-3.1-nemotron-70b-instruct",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.5,
      top_p: 1,
      max_tokens: 1024,
    })

    res.status(200).json({ response: completion.choices[0].message.content })
  } catch (error) {
    console.error('Error generating NVIDIA response:', error)
    res.status(500).json({ error: 'Error generating response' })
  }
}

