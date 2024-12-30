"use client"

import React, { useState, useEffect, useRef, useMemo } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Html, useTexture, Stars, Trail, useGLTF } from '@react-three/drei'
import { EffectComposer, SSR, Bloom, DepthOfField, Noise, N8AO, TiltShift, ChromaticAberration } from '@react-three/postprocessing'
import { KernelSize, BlendFunction } from 'postprocessing'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Download, Upload, Brain, Network, Database, FileText } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { HfInference } from '@huggingface/inference'
import axios from 'axios'
import * as pdfjs from 'pdfjs-dist'
import { v4 as uuidv4 } from 'uuid'
import Peer from 'peerjs'
import localforage from 'localforage'
import { create } from 'zustand'
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer'

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

// Enhanced Optical Raytracer
class OpticalRaytracer {
  constructor() {
    this.maxBounces = 8
    this.samplesPerPixel = 64
    this.wavelengths = [380, 550, 780] // nm
  }

  traceRay(ray, medium) {
    let energy = 1.0
    let currentRay = ray
    let result = new THREE.Vector3()

    for (let bounce = 0; bounce < this.maxBounces; bounce++) {
      const intersection = this.findIntersection(currentRay)
      if (!intersection) break

      const [scattered, probability] = this.computeScattering(currentRay, intersection, medium)

      energy *= probability
      if (energy < 0.001) break

      result.add(this.evaluateSpectrum(scattered, this.wavelengths))
      currentRay = scattered
    }

    return result
  }

  computeScattering(ray, intersection, medium) {
    const n1 = medium.refractiveIndex
    const n2 = intersection.material.refractiveIndex
    
    const cosTheta = Math.min(ray.direction.dot(intersection.normal.negate()), 1.0)
    const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta)
    const sinThetaT = n1 / n2 * sinTheta

    if (sinThetaT >= 1.0) {
      return [this.reflect(ray, intersection), 1.0]
    }

    const cosThetaT = Math.sqrt(1.0 - sinThetaT * sinThetaT)
    const Rs = ((n1 * cosTheta - n2 * cosThetaT) / (n1 * cosTheta + n2 * cosThetaT)) ** 2
    const Rp = ((n1 * cosThetaT - n2 * cosTheta) / (n1 * cosThetaT + n2 * cosTheta)) ** 2
    const Fr = (Rs + Rp) / 2

    if (Math.random() < Fr) {
      return [this.reflect(ray, intersection), Fr]
    } else {
      return [this.refract(ray, intersection, n1/n2), 1-Fr]
    }
  }

  // Implement other methods: findIntersection, evaluateSpectrum, reflect, refract
}

// Improved Holographic System
class HolographicPlate {
  constructor(resolution) {
    this.resolution = resolution
    this.plate = new Float32Array(resolution * resolution * 2)
    this.referenceWave = this.createReferenceWave()
  }

  record(objectWave) {
    const interference = new Float32Array(this.resolution * this.resolution * 2)
    
    for (let y = 0; y < this.resolution; y++) {
      for (let x = 0; x < this.resolution; x++) {
        const i = (y * this.resolution + x) * 2
        
        const or = objectWave[i]
        const oi = objectWave[i + 1]
        const rr = this.referenceWave[i]
        const ri = this.referenceWave[i + 1]

        interference[i] = or * rr - oi * ri
        interference[i + 1] = or * ri + oi * rr
      }
    }

    for (let i = 0; i < this.plate.length; i++) {
      this.plate[i] += interference[i]
    }
  }

  reconstruct(illuminationWave) {
    const reconstruction = new Float32Array(this.resolution * this.resolution * 2)
    
    for (let y = 0; y < this.resolution; y++) {
      for (let x = 0; x < this.resolution; x++) {
        const i = (y * this.resolution + x) * 2
        
        const hr = this.plate[i]
        const hi = this.plate[i + 1]
        const ir = illuminationWave[i]
        const ii = illuminationWave[i + 1]

        reconstruction[i] = hr * ir - hi * ii
        reconstruction[i + 1] = hr * ii + hi * ir
      }
    }

    return reconstruction
  }

  createReferenceWave() {
    // Implement reference wave creation
    return new Float32Array(this.resolution * this.resolution * 2)
  }
}

// Enhanced Holographic Neural Memory
class HolographicMemory {
  constructor(dimensions) {
    this.dimensions = dimensions
    this.hologramPlates = new Map()
    this.phaseConjugator = new PhaseConjugator()
  }

  store(pattern, association) {
    const objectWave = this.patternToWave(pattern)
    
    if (!this.hologramPlates.has(pattern)) {
      this.hologramPlates.set(pattern, new HolographicPlate(this.dimensions))
    }

    const plate = this.hologramPlates.get(pattern)
    const associationWave = this.patternToWave(association)
    plate.record(associationWave)
  }

  recall(pattern) {
    const plate = this.hologramPlates.get(pattern)
    if (!plate) return null

    const probeWave = this.patternToWave(pattern)
    const conjugateWave = this.phaseConjugator.conjugate(probeWave)
    const reconstruction = plate.reconstruct(conjugateWave)

    return this.waveToPattern(reconstruction)
  }

  patternToWave(pattern) {
    const wave = new Float32Array(this.dimensions * this.dimensions * 2)
    // Implement pattern to wave transformation
    return wave
  }

  waveToPattern(wave) {
    // Implement wave to pattern transformation
    return "reconstructed pattern"
  }
}

class PhaseConjugator {
  conjugate(wave) {
    const conjugatedWave = new Float32Array(wave.length)
    for (let i = 0; i < wave.length; i += 2) {
      conjugatedWave[i] = wave[i]
      conjugatedWave[i + 1] = -wave[i + 1]
    }
    return conjugatedWave
  }
}

// Continuous Learning System
class ContinuousLearningSystem {
  constructor() {
    this.memory = new HolographicMemory(1024)
    this.raytracer = new OpticalRaytracer()
    this.plasticityFactor = 0.1
  }

  learn(input, output) {
    const inputPattern = this.encodePattern(input)
    const outputPattern = this.encodePattern(output)

    this.memory.store(inputPattern, outputPattern)
    this.updatePlasticity(inputPattern, outputPattern)
  }

  recall(input) {
    const pattern = this.encodePattern(input)
    const reconstruction = this.memory.recall(pattern)
    return this.decodePattern(reconstruction)
  }

  updatePlasticity(input, output) {
    const timeDiff = this.computeTimingDifference(input, output)
    const plasticityChange = this.stdpCurve(timeDiff)
    this.plasticityFactor *= (1 + plasticityChange)
  }

  encodePattern(input) {
    // Implement pattern encoding
    return "encoded pattern"
  }

  decodePattern(pattern) {
    // Implement pattern decoding
    return "decoded pattern"
  }

  computeTimingDifference(input, output) {
    // Implement timing difference computation
    return 0
  }

  stdpCurve(timeDiff) {
    // Implement STDP curve
    return 0
  }
}

// Enhanced shader code for optical simulation
const opticalVertexShader = `
  uniform float time;
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    vec3 pos = position;
    
    // Add wave-like movement
    pos.y += sin(pos.x * 2.0 + time) * 0.02;
    pos.z += cos(pos.y * 2.0 + time) * 0.02;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`

const opticalFragmentShader = `
  uniform float time;
  uniform vec3 color;
  uniform sampler2D hologramTexture;
  uniform float interference;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  float fresnel(vec3 viewDir, vec3 normal, float power) {
    return pow(1.0 - abs(dot(viewDir, normal)), power);
  }
  
  void main() {
    vec3 viewDir = normalize(-vPosition);
    float fresnelTerm = fresnel(viewDir, vNormal, 3.0);
    
    // Create interference patterns
    float pattern = sin(vUv.x * 50.0 + time) * cos(vUv.y * 50.0 - time);
    pattern *= interference;
    
    // Holographic effect
    vec4 hologram = texture2D(hologramTexture, vUv + vec2(sin(time * 0.5) * 0.1));
    
    // Combine effects
    vec3 finalColor = mix(color, hologram.rgb, 0.5) + pattern * 0.2;
    finalColor += fresnelTerm * vec3(0.5, 0.7, 1.0);
    
    gl_FragColor = vec4(finalColor, 0.85);
  }
`

// Zustand store with enhanced state management
const useStore = create((set, get) => ({
  neurons: [],
  connections: [],
  contextNodes: [],
  waveStates: [],
  interferencePatterns: [],
  learningSystem: new ContinuousLearningSystem(),
  setNeurons: (neurons) => set({ neurons }),
  setConnections: (connections) => set({ connections }),
  setContextNodes: (contextNodes) => set({ contextNodes }),
  setWaveStates: (waveStates) => set({ waveStates }),
  setInterferencePatterns: (patterns) => set({ interferencePatterns: patterns }),
  updateWaveState: (index, newState) => {
    const waveStates = [...get().waveStates]
    waveStates[index] = newState
    set({ waveStates })
  },
  learn: (input, output) => {
    get().learningSystem.learn(input, output)
    // Update visualization state here
  },
  recall: (input) => {
    const output = get().learningSystem.recall(input)
    // Update visualization state here
    return output
  }
}))

// Enhanced 3D visualization components
function OpticalNeuron({ position, activation, relevance, isContextNode }) {
  const meshRef = useRef()
  const [hovered, setHovered] = useState(false)
  const hologramTexture = useTexture('/placeholder.svg?height=128&width=128')
  
  const materialRef = useRef()
  const waveRef = useRef({
    time: 0,
    amplitude: Math.random() * 0.1 + 0.05,
    frequency: Math.random() * 2 + 1
  })
  
  const color = useMemo(() => (
    isContextNode 
      ? new THREE.Color(relevance, 0, 1 - relevance)
      : new THREE.Color(activation, 0, 1 - activation)
  ), [isContextNode, relevance, activation])

  useFrame((state) => {
    if (meshRef.current && materialRef.current) {
      waveRef.current.time += 0.05
      const waveOffset = Math.sin(waveRef.current.time * waveRef.current.frequency) * waveRef.current.amplitude
      meshRef.current.position.y = position[1] + waveOffset
      
      materialRef.current.uniforms.time.value = state.clock.elapsedTime
      materialRef.current.uniforms.interference.value = 0.5 + Math.sin(state.clock.elapsedTime) * 0.5
    }
  })

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[isContextNode ? 0.2 : 0.15, 32, 32]} />
        <shaderMaterial
          ref={materialRef}
          vertexShader={opticalVertexShader}
          
          fragmentShader={opticalFragmentShader}
          uniforms={{
            time: { value: 0 },
            color: { value: color },
            hologramTexture: { value: hologramTexture },
            interference: { value: 0.5 }
          }}
          transparent
        />
      </mesh>
      <Trail
        width={0.05}
        length={5}
        color={color}
        attenuation={(t) => t * t}
      >
        <mesh>
          <sphereGeometry args={[0.02, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>
      </Trail>
      {hovered && (
        <Html distanceFactor={10}>
          <div className="bg-black bg-opacity-75 text-white p-2 rounded text-xs">
            {isContextNode ? `Relevance: ${relevance.toFixed(2)}` : `Activation: ${activation.toFixed(2)}`}
          </div>
        </Html>
      )}
    </group>
  )
}

function OpticalConnection({ start, end, strength }) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end])

  const curve = useMemo(() => new THREE.CatmullRomCurve3(points), [points])

  return (
    <mesh>
      <tubeGeometry args={[curve, 64, 0.01, 8, false]} />
      <meshBasicMaterial color={new THREE.Color(strength, 0, 1 - strength)} transparent opacity={0.3} />
    </mesh>
  )
}

function NeuralNetwork() {
  const { neurons, connections, contextNodes } = useStore()

  return (
    <group>
      {neurons.map((neuron, i) => (
        <OpticalNeuron key={i} position={neuron.position} activation={neuron.activation} isContextNode={false} />
      ))}
      {contextNodes.map((node, i) => (
        <OpticalNeuron key={`context-${i}`} position={node.position} relevance={node.relevance} isContextNode={true} />
      ))}
      {connections.map((connection, i) => (
        <OpticalConnection
          key={i}
          start={neurons[connection.start].position}
          end={neurons[connection.end].position}
          strength={connection.strength}
        />
      ))}
    </group>
  )
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.1} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      <NeuralNetwork />
      <EffectComposer>
        <SSR intensity={0.45} exponent={1} distance={10} fade={10} roughnessFade={1} thickness={10} ior={0.45} maxRoughness={1} maxDepthDifference={10} blend={0.95} correction={1} correctionRadius={1} blur={0} blurKernel={1} blurSharpness={10} jitter={0.75} jitterRoughness={0.2} steps={40} refineSteps={5} missedRays={true} useNormalMap={true} useRoughnessMap={true} resolutionScale={1} velocityResolutionScale={1} />
        <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} />
        <Noise opacity={0.02} />
        <DepthOfField focusDistance={0} focalLength={0.02} bokehScale={2} height={480} />
        <N8AO aoRadius={1} intensity={1} />
        <TiltShift blur={0.5} />
        <ChromaticAberration offset={[0.001, 0.001]} />
      </EffectComposer>
      <OrbitControls />
    </>
  )
}

export default function EnhancedHolographicNeuralNetworkApp() {
  const [chatHistory, setChatHistory] = useState([])
  const [inputText, setInputText] = useState('')
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
  const [llmModel, setLLMModel] = useState(null)
  const [isLLMActive, setIsLLMActive] = useState(false)
  const { setNeurons, setConnections, setContextNodes, learn, recall } = useStore()

  useEffect(() => {
    // Initialize neurons, connections, and context nodes
    setNeurons(Array.from({ length: 100 }, () => ({
      position: [
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ],
      activation: Math.random()
    })))
    setConnections(Array.from({ length: 200 }, () => ({
      start: Math.floor(Math.random() * 100),
      end: Math.floor(Math.random() * 100),
      strength: Math.random()
    })))
    setContextNodes(Array.from({ length: 20 }, () => ({
      position: [
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ],
      relevance: Math.random()
    })))

    const hf = new HfInference("You API key")
    setLLMModel(hf)
    setIsLLMActive(true)

    createAccountAutomatically()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputText.trim()) return

    let response
    if (isLLMActive && llmModel) {
      try {
        const llmResponse = await llmModel.textGeneration({
          model: 'facebook/opt-350m',
          inputs: inputText,
          parameters: {
            max_new_tokens: 50,
            temperature: 0.7,
            top_p: 0.95,
            repetition_penalty: 1.1,
          },
        })
        response = llmResponse.generated_text
      } catch (error) {
        console.error('Error generating response:', error)
        response = "Sorry, I couldn't generate a response."
      }
    } else {
      response = recall(inputText)
    }

    learn(inputText, response)
    setChatHistory(prev => [...prev, { type: 'user', text: inputText }, { type: 'bot', text: response }])
    setInputText('')

    if (p2pConnected) {
      try {
        await axios.post('/api/store', { key: inputText, value: response })
      } catch (error) {
        console.error('Error storing data in P2P network:', error)
      }
    }

    setPredictedWords(response.split(' ').slice(0, 10))
  }

  const createAccountAutomatically = async () => {
    try {
      const generatedNodeId = uuidv4()
      const response = await axios.post('/api/create-account', { nodeId: generatedNodeId })
      console.log('Account created automatically:', response.data)
      setNodeId(response.data.nodeId)
      await handleConnectP2P()
    } catch (error) {
      console.error('Error creating account automatically:', error)
      setError('Failed to create account automatically')
    }
  }

  const handleConnectP2P = async () => {
    try {
      await axios.post('/api/connect')
      setP2pConnected(true)
      console.log('Connected to P2P network')
    } catch (error) {
      console.error('Error connecting to P2P network:', error)
      setError('Failed to connect to P2P network')
    }
  }

  const handleLearn = () => {
    if (learnInput.trim() && learnResponse.trim()) {
      learn(learnInput, learnResponse)
      setLearnInput('')
      setLearnResponse('')
      alert('Learning completed')
    }
  }

  const handleSave = () => {
    const knowledge = JSON.stringify(useStore.getState())
    const blob = new Blob([knowledge], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'enhanced_holographic_nn_knowledge.json'
    a.click()
  }

  const handleLoad = async (event) => {
    const file = event.target.files?.[0]
    if (file) {
      const knowledge = await file.text()
      try {
        const state = JSON.parse(knowledge)
        useStore.setState(state)
        alert('Knowledge loaded successfully')
      } catch (error) {
        console.error("Error importing knowledge:", error)
        alert('Error loading knowledge')
      }
    }
  }

  const handleTrain = () => {
    setIsTraining(true)
    setTrainingProgress(0)
    const trainStep = (i) => {
      if (i >= 100) {
        setIsTraining(false)
        setTrainingProgress(0)
        alert('Training completed successfully')
        return
      }

      const input = `Random input ${i}`
      const output = `Random output ${i}`
      learn(input, output)
      setTrainingProgress(Math.round(((i + 1) / 100) * 100))

      setTimeout(() => trainStep(i + 1), 100)
    }

    trainStep(0)
  }

  const processPDF = async (file) => {
    const arrayBuffer = await file.arrayBuffer()
    const pdf = await pdfjs.getDocument(arrayBuffer).promise
    const totalPages = pdf.numPages
    let processedText = ''
    let currentPage = 1

    const processNextChunk = async () => {
      if (currentPage > totalPages) {
        setIsProcessing(false)
        setProgress(100)
        return processedText
      }

      const endPage = Math.min(currentPage + 4, totalPages)
      let chunkText = ''

      for (let i = currentPage; i <= endPage; i++) {
        const page = await pdf.getPage(i)
        const content = await page.getTextContent()
        chunkText += content.items.map((item) => item.str).join(' ') + ' '
        setProgress((i / totalPages) * 100)
      }

      processedText += chunkText
      currentPage = endPage + 1

      // Process the chunk
      const words = chunkText.toLowerCase().match(/\b\w+\b/g) || []
      for (let i = 0; i < words.length - 1; i++) {
        learn(words[i], words[i + 1])
      }

      // Schedule the next chunk
      setTimeout(processNextChunk, 0)
    }

    // Start processing
    await processNextChunk()
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
          const words = text.toLowerCase().match(/\b\w+\b/g) || []
          for (let i = 0; i < words.length - 1; i++) {
            learn(words[i], words[i + 1])
          }
        } else if (file.name.endsWith('.pdf')) {
          await processPDF(file)
        } else {
          throw new Error('Unsupported file type')
        }

        setError(null)
        alert('File processed successfully')
      } catch (err) {
        setError('Error processing file. Please try again.')
        setIsProcessing(false)
      }
    }
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <div className="w-1/2 p-4 overflow-y-auto">
        <Tabs defaultValue="chat">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="learn">Learn</TabsTrigger>
            <TabsTrigger value="manage">Manage</TabsTrigger>
            <TabsTrigger value="network">Network</TabsTrigger>
          </TabsList>
          <TabsContent value="chat">
            <Card>
              <CardHeader>
                <CardTitle>Enhanced Holographic Neural Network Chat</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 mb-4 h-96 overflow-y-auto">
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
          </TabsContent>
          <TabsContent value="learn">
            <Card>
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
          </TabsContent>
          <TabsContent value="manage">
            <Card>
              <CardHeader>
                <CardTitle>Knowledge Management</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex space-x-2 mb-4">
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
                <div>
                  <Button onClick={handleTrain} disabled={isTraining}>
                    {isTraining ? 'Training...' : 'Train'}
                  </Button>
                  {isTraining && (
                    <div className="mt-2">
                      <Progress value={trainingProgress} className="w-full" />
                      <p className="text-center mt-2">{trainingProgress}% completed</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
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
          </TabsContent>
          <TabsContent value="network">
            <Card>
              <CardHeader>
                <CardTitle>P2P Network</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Input
                    type="text"
                    value={nodeId}
                    placeholder="Node ID"
                    disabled
                  />
                  <Button onClick={handleConnectP2P} disabled={p2pConnected}>
                    {p2pConnected ? 'Connected' : 'Connect to P2P Network'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>
      <div className="w-1/2">
        <Canvas>
          <Scene />
        </Canvas>
      </div>
    </div>
  )
}
