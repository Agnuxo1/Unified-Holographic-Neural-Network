"use client"

import React, { useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { useTexture } from '@react-three/drei'

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

export function HolographicPlane() {
  const shaderRef = useRef()
  const texture = useTexture('/placeholder.svg?height=1024&width=1024')

  useFrame(({ clock }) => {
    if (shaderRef.current) {
      shaderRef.current.uniforms.time.value = clock.getElapsedTime()
    }
  })

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -10, 0]}>
      <planeGeometry args={[100, 100]} />
      <shaderMaterial
        ref={shaderRef}
        vertexShader={holographicVertexShader}
        fragmentShader={holographicFragmentShader}
        uniforms={{
          time: { value: 0 },
          color: { value: new THREE.Color(0.1, 0.3, 0.6) },
          hologramTexture: { value: texture }
        }}
        transparent
      />
    </mesh>
  )
}

