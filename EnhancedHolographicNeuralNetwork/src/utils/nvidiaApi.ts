export async function generateNvidiaResponse(prompt: string): Promise<string> {
  try {
    const response = await fetch('/api/nvidia-completion', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    })

    if (!response.ok) {
      throw new Error('Failed to generate response')
    }

    const data = await response.json()
    return data.response
  } catch (error) {
    console.error('Error generating NVIDIA response:', error)
    throw error
  }
}

