"use client"

import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import * as pdfjs from 'pdfjs-dist'
import { v4 as uuidv4 } from 'uuid'
import Peer from 'peerjs'
import localforage from 'localforage'

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Download, Upload } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

import { EnhancedHolographicNeuralNetwork } from '@/lib/EnhancedHolographicNeuralNetwork'
import { Scene } from '@/components/HolographicNeuralNetwork/Scene'
import Chat from '@/components/Chat'
import Learning from '@/components/Learning'
import KnowledgeManagement from '@/components/KnowledgeManagement'
import Training from '@/components/Training'
import FileProcessing from '@/components/FileProcessing'
import P2PNetwork from '@/components/P2PNetwork'

import { LLAMA_INDEX_API, NEMOTRON_70B_API, RAG_NVIDIA_API, NEMO_GUARDRAILS_API, useNvidiaAPIs } from '@/src/config'
import { generateNvidiaResponse } from '@/src/utils/nvidiaApi'

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

export default function EnhancedUnifiedHolographicNeuralNetwork() {
  const [chatHistory, setChatHistory] = useState([])
  const [inputText, setInputText] = useState('')
  const [hnn, setHnn] = useState(null)
  const [isLLMActive, setIsLLMActive] = useState(true)
  const [error, setError] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [learnInput, setLearnInput] = useState('')
  const [learnResponse, setLearnResponse] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [predictedWords, setPredictedWords] = useState([])
  const [nodeId, setNodeId] = useState('')
  const [p2pConnected, setP2pConnected] = useState(false)
  const [connectedPeers, setConnectedPeers] = useState([])
  const peerRef = useRef(null)

  useEffect(() => {
    const initializeHNN = async () => {
      let storedHNN = await localforage.getItem('hnn')
      if (storedHNN) {
        const newHnn = new EnhancedHolographicNeuralNetwork(100)
        newHnn.importKnowledge(storedHNN)
        setHnn(newHnn)
      } else {
        setHnn(new EnhancedHolographicNeuralNetwork(100))
      }
    }

    initializeHNN()
    initializeP2P()
  }, [])

  useEffect(() => {
    if (hnn) {
      const saveHNN = async () => {
        await localforage.setItem('hnn', hnn.exportKnowledge())
      }
      saveHNN()
    }
  }, [hnn])

  const initializeP2P = () => {
    const peer = new Peer(uuidv4())
    peerRef.current = peer

    peer.on('open', (id) => {
      setNodeId(id)
      setP2pConnected(true)
    })

    peer.on('connection', (conn) => {
      conn.on('data', (data) => {
        handlePeerData(data)
      })
      setConnectedPeers(prev => [...prev, conn.peer])
    })

    peer.on('error', (error) => {
      console.error('P2P Error:', error)
      setError('P2P connection error')
    })
  }

  const handlePeerData = (data) => {
    if (data.type === 'knowledge') {
      hnn.importKnowledge(data.knowledge)
    }
  }

  const broadcastKnowledge = () => {
    const knowledge = hnn.exportKnowledge()
    connectedPeers.forEach(peerId => {
      const conn = peerRef.current.connect(peerId)
      conn.on('open', () => {
        conn.send({ type: 'knowledge', knowledge })
      })
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputText.trim() || !hnn) return

    let response
    let ragContext = []
    if (isLLMActive) {
      try {
        if (useNvidiaAPIs) {
          const llamaIndexResult = await hnn.useLlamaIndex(inputText)
          const nemotronResult = await hnn.useNemotron70B(inputText)
          const ragResult = await hnn.useRagNvidia(inputText)
          const guardrailsResult = await hnn.useNemoGuardrails(inputText)

          response = `LlamaIndex: ${llamaIndexResult}
Nemotron: ${nemotronResult}
RAG: ${ragResult}
Guardrails: ${guardrailsResult}`
        } else {
          ragContext = [
            { text: "Relevant context 1", score: 0.9 },
            { text: "Relevant context 2", score: 0.7 },
            { text: "Relevant context 3", score: 0.5 }
          ]
          hnn.updateContextNodes(ragContext)

          response = await generateNvidiaResponse(inputText)
        }
      } catch (error) {
        console.error('Error generating response:', error)
        response = "Sorry, I couldn't generate a response."
      }
    } else {
      response = hnn.generateResponse(inputText)
    }

    hnn.learn(inputText, response)
    setChatHistory(prev => [...prev, { type: 'user', text: inputText }, { type: 'bot', text: response }])
    
    setInputText('')

    // Update predicted words
    const words = hnn.generateWords(inputText.split(' ').pop(), 10)
    setPredictedWords(words)

    // Broadcast updated knowledge to peers
    broadcastKnowledge()
  }

  const handleLearn = () => {
    if (learnInput.trim() && learnResponse.trim() && hnn) {
      hnn.learn(learnInput, learnResponse)
      setLearnInput('')
      setLearnResponse('')
      alert('Learning completed')
      broadcastKnowledge()
    }
  }

  const handleSave = () => {
    if (hnn) {
      const knowledge = hnn.exportKnowledge()
      const blob = new Blob([knowledge], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'holographic_nn_knowledge.json'
      a.click()
    }
  }

  const handleLoad = (event) => {
    const file = event.target.files?.[0]
    if (file && hnn) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const knowledge = e.target?.result
        if (typeof knowledge === 'string') {
          const success = hnn.importKnowledge(knowledge)
          if (success) {
            alert('Knowledge loaded successfully')
            broadcastKnowledge()
          } else {
            alert('Error loading knowledge')
          }
        }
      }
      reader.readAsText(file)
    }
  }

  const handleTrain = () => {
    setIsTraining(true)
    setTrainingProgress(0)
    const trainStep = (i) => {
      if (i >= 100) {  // Train on 100 random samples
        setIsTraining(false)
        setTrainingProgress(0)
        alert('Training completed successfully')
        broadcastKnowledge()
        return
      }

      const input = `Random input ${i}`
      const output = `Random output ${i}`
      hnn.learn(input, output)
      setTrainingProgress(Math.round(((i + 1) / 100) * 100))

      setTimeout(() => trainStep(i + 1), 100)
    }

    trainStep(0)
  }

  const processPDF = async (file) => {
    const arrayBuffer = await file.arrayBuffer()
    const pdf = await pdfjs.getDocument(arrayBuffer).promise
    let text = ''
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i)
      const content = await page.getTextContent()
      text += content.items.map((item) => item.str).join(' ') + ' '
      setProgress((i / pdf.numPages) * 100)
    }
    return text
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]
    if (file) {
      setIsProcessing(true)
      setProgress(0)
      try {
        let text = ''
        if (file.name.endsWith('.txt')) {
          text = await file.text()
        } else if (file.name.endsWith('.pdf')) {
          text = await processPDF(file)
        } else {
          throw new Error('Unsupported file type')
        }

        const words = text.toLowerCase().match(/\b\w+\b/g) || []
        for (let i = 0; i < words.length - 1; i++) {
          hnn.learn(words[i], words[i + 1])
        }

        setIsProcessing(false)
        setError(null)
        alert('File processed successfully')
        broadcastKnowledge()
      } catch (err) {
        setError('Error processing file. Please try again.')
        setIsProcessing(false)
      }
    }
  }

  return (
    <div className="flex h-screen">
      <div className="w-1/2 p-4 overflow-y-auto">
        <Chat 
          chatHistory={chatHistory}
          inputText={inputText}
          setInputText={setInputText}
          handleSubmit={handleSubmit}
          predictedWords={predictedWords}
        />
        <P2PNetwork
          nodeId={nodeId}
          p2pConnected={p2pConnected}
          connectedPeers={connectedPeers}
        />
        <Learning
          learnInput={learnInput}
          setLearnInput={setLearnInput}
          learnResponse={learnResponse}
          setLearnResponse={setLearnResponse}
          handleLearn={handleLearn}
        />
        <KnowledgeManagement
          handleSave={handleSave}
          handleLoad={handleLoad}
        />
        <Training
          isTraining={isTraining}
          trainingProgress={trainingProgress}
          handleTrain={handleTrain}
        />
        <FileProcessing
          handleFileUpload={handleFileUpload}
          isProcessing={isProcessing}
          progress={progress}
        />
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>
      <div className="w-1/2">
        <Scene 
          neurons={hnn ? hnn.neurons : []} 
          connections={hnn ? hnn.connections : []} 
          contextNodes={hnn ? hnn.contextNodes : []}
        />
      </div>
    </div>
  )
}

