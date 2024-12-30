"use client"

import React, { useRef, useState } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Html, Trail, useTexture } from '@react-three/drei'

const holographicVertexShader = `
varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
void main() {
  vUv = uv;
  vPosition = position;
  vNormal = normal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const holographicFragmentShader = `
uniform float time;
uniform vec3 color;
uniform sampler2D hologramTexture;
varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;

float fresnel(vec3 normal, vec3 viewDir) {
  return pow(1.0 + dot(normal, viewDir), 3.0);
}

void main() {
  vec3 viewDir = normalize(cameraPosition - vPosition);
  float fresnelTerm = fresnel(vNormal, viewDir);
  
  vec2 uv = vUv + 0.1 * vec2(sin(time + vPosition.x * 10.0), cos(time + vPosition.y * 10.0));
  vec3 hologram = texture2D(hologramTexture, uv).rgb;
  
  vec3 finalColor = mix(color, hologram, 0.5) + fresnelTerm * vec3(0.1, 0.3, 0.5);
  float alpha = 0.7 + 0.3 * sin(time * 2.0 + vPosition.z * 5.0);
  
  gl_FragColor = vec4(finalColor, alpha);
}
`

export function Neuron({ position, activation, relevance, isContextNode }) {
  const meshRef = useRef()
  const [hovered, setHovered] = useState(false)
  const hologramTexture = useTexture('/placeholder.svg?height=128&width=128')

  const color = isContextNode 
    ? new THREE.Color(relevance, 0, 1 - relevance)
    : new THREE.Color(activation, 0, 1 - activation)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.2
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.2
      meshRef.current.material.uniforms.time.value = state.clock.elapsedTime
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
          vertexShader={holographicVertexShader}
          fragmentShader={holographicFragmentShader}
          uniforms={{
            time: { value: 0 },
            color: { value: color },
            hologramTexture: { value: hologramTexture }
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

