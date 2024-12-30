"use client"

import React, { useEffect } from 'react'
import * as THREE from 'three'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import { EffectComposer, SSR, Bloom, DepthOfField, Noise } from '@react-three/postprocessing'
import { KernelSize } from 'postprocessing'
import { Neuron } from './Neuron'
import { Connection } from './Connection'
import { HolographicPlane } from './HolographicPlane'

interface SceneProps {
  neurons: any[]
  connections: any[]
  contextNodes: any[]
}

export function Scene({ neurons, connections, contextNodes }: SceneProps) {
  return (
    <Canvas>
      <ambientLight intensity={0.1} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      <NeuralNetwork neurons={neurons} connections={connections} contextNodes={contextNodes} />
      <HolographicPlane />
      <EffectComposer>
        <SSR intensity={0.45} exponent={1} distance={10} fade={10} roughnessFade={1} thickness={10} ior={0.45} maxRoughness={1} maxDepthDifference={10} blend={0.95} correction={1} correctionRadius={1} blur={0} blurKernel={1} blurSharpness={10} jitter={0.75} jitterRoughness={0.2} steps={40} refineSteps={5} missedRays={true} useNormalMap={true} useRoughnessMap={true} resolutionScale={1} velocityResolutionScale={1} />
        <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} />
        <Noise opacity={0.02} />
        <DepthOfField focusDistance={0} focalLength={0.02} bokehScale={2} height={480} />
      </EffectComposer>
      <CameraSetup />
      <OrbitControls />
    </Canvas>
  )
}

interface NeuralNetworkProps {
  neurons: any[]
  connections: any[]
  contextNodes: any[]
}

function NeuralNetwork({ neurons, connections, contextNodes }: NeuralNetworkProps) {
  return (
    <group>
      {neurons.map((neuron, i) => (
        <Neuron key={i} position={neuron.position} activation={neuron.activation} isContextNode={false} />
      ))}
      {contextNodes.map((node, i) => (
        <Neuron key={`context-${i}`} position={node.position} relevance={node.relevance} isContextNode={true} />
      ))}
      {connections.map((connection, i) => (
        <Connection
          key={i}
          start={neurons[connection.start].position}
          end={neurons[connection.end].position}
          strength={connection.strength}
        />
      ))}
    </group>
  )
}

function CameraSetup() {
  const { camera } = useThree()

  useEffect(() => {
    camera.position.set(0, 20, 40)
  }, [camera])

  return null
}

