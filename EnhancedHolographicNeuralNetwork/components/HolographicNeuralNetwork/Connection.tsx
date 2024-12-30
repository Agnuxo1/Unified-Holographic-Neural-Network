import React, { useMemo } from 'react'
import * as THREE from 'three'

interface ConnectionProps {
  start: [number, number, number]
  end: [number, number, number]
  strength: number
}

export function Connection({ start, end, strength }: ConnectionProps) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end])

  return (
    <mesh>
      <tubeGeometry args={[new THREE.CatmullRomCurve3(points), 64, 0.01, 8, false]} />
      <meshBasicMaterial color={new THREE.Color(strength, 0, 1 - strength)} transparent opacity={0.3} />
    </mesh>
  )
}

