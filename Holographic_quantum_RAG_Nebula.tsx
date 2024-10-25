'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Brain, MessageSquare, FileQuestion, Save, Database } from 'lucide-react';

interface WordNode {
  word: string;
  count: number;
  next: Map<string, { word: string; count: number }>;
  color: {
    hue: number;
    saturation: number;
    brightness: number;
    alpha: number;
  };
  documents: Set<string>;
  lastAccessed: number;
  strength: number;
}

interface Memory {
  nodes: Map<string, WordNode>;
  totalDocuments: number;
  lastConsolidation: number;
}

const vertexShader = `
attribute vec4 a_position;
varying vec2 v_texcoord;
varying float v_depth;

void main() {
  gl_Position = a_position;
  v_texcoord = a_position.xy * 0.5 + 0.5;
  v_depth = a_position.z;
}`;

const fragmentShader = `
precision highp float;
varying vec2 v_texcoord;
varying float v_depth;
uniform float u_time;
uniform vec4 u_wordColors[128];
uniform vec2 u_wordPositions[128];
uniform float u_wordStrengths[128];
uniform float u_wordCount;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  vec2 uv = v_texcoord;
  float time = u_time * 0.001;
  vec4 finalColor = vec4(0.0);
  
  for(float i = 0.0; i < 128.0; i++) {
    if(i >= u_wordCount) break;
    
    vec2 wordPos = u_wordPositions[int(i)];
    vec4 wordColor = u_wordColors[int(i)];
    float strength = u_wordStrengths[int(i)];
    
    float dist = distance(uv, wordPos);
    float wave = sin(dist * 50.0 - time * 2.0) * 0.5 + 0.5;
    vec3 rgb = hsv2rgb(vec3(wordColor.x, wordColor.y, wordColor.z));
    float alpha = wordColor.w * strength;
    
    float pulse = sin(time * strength * 3.0) * 0.2 + 0.8;
    finalColor += vec4(rgb * wave * pulse, alpha) * (1.0 - smoothstep(0.0, 0.3, dist));
  }
  
  finalColor += vec4(0.1, 0.1, 0.2, 0.1);
  finalColor = clamp(finalColor, 0.0, 1.0);
  gl_FragColor = finalColor;
}`;

const MEMORY_KEY = 'neural_holographic_memory';
const CONSOLIDATION_INTERVAL = 1000 * 60 * 60;
const MEMORY_DECAY_RATE = 0.1;
const MAX_MEMORY_STRENGTH = 5;

export default function HolographicMemoryProcessor() {
  const [memory, setMemory] = useState<Memory>({
    nodes: new Map(),
    totalDocuments: 0,
    lastConsolidation: Date.now()
  });
  const [inputText, setInputText] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({ totalWords: 0, activeMemories: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const rafRef = useRef<number>();

  useEffect(() => {
    const savedMemory = localStorage.getItem(MEMORY_KEY);
    if (savedMemory) {
      try {
        const parsed = JSON.parse(savedMemory);
        const reconstructed: Memory = {
          nodes: new Map(),
          totalDocuments: parsed.totalDocuments,
          lastConsolidation: parsed.lastConsolidation
        };
        
        Object.entries(parsed.nodes).forEach(([word, node]: [string, any]) => {
          reconstructed.nodes.set(word, {
            ...node,
            next: new Map(Object.entries(node.next)),
            documents: new Set(node.documents)
          });
        });
        
        setMemory(reconstructed);
      } catch (error) {
        console.error('Error loading memory:', error);
      }
    }
  }, []);

  const saveMemory = useCallback(() => {
    const serializable = {
      nodes: Object.fromEntries(memory.nodes),
      totalDocuments: memory.totalDocuments,
      lastConsolidation: memory.lastConsolidation
    };
    localStorage.setItem(MEMORY_KEY, JSON.stringify(serializable));
  }, [memory]);

  const consolidateMemory = useCallback(() => {
    const now = Date.now();
    if (now - memory.lastConsolidation < CONSOLIDATION_INTERVAL) return;

    const newNodes = new Map(memory.nodes);
    let activeMemories = 0;

    for (const [word, node] of newNodes.entries()) {
      const timeDiff = (now - node.lastAccessed) / (1000 * 60 * 60 * 24);
      const decay = Math.exp(-MEMORY_DECAY_RATE * timeDiff);
      node.strength *= decay;

      if (node.strength < 0.1) {
        newNodes.delete(word);
      } else {
        activeMemories++;
      }
    }

    setMemory(prev => ({
      ...prev,
      nodes: newNodes,
      lastConsolidation: now
    }));

    setStats(prev => ({
      ...prev,
      activeMemories
    }));

    saveMemory();
  }, [memory, saveMemory]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setLoading(true);
    setProgress(0);
    
    try {
      const text = await file.text();
      const documentId = `doc_${Date.now()}`;
      processText(text, documentId);
      setProgress(100);
    } catch (error) {
      console.error('Error processing file:', error);
    } finally {
      setLoading(false);
    }
  };

  const processText = useCallback((text: string, documentId: string) => {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const newNodes = new Map(memory.nodes);
    
    for (let i = 0; i < words.length - 1; i++) {
      const currentWord = words[i];
      const nextWord = words[i + 1];
      
      if (!newNodes.has(currentWord)) {
        newNodes.set(currentWord, {
          word: currentWord,
          count: 1,
          next: new Map([[nextWord, { word: nextWord, count: 1 }]]),
          color: {
            hue: Math.random() * 360,
            saturation: 50,
            brightness: 50,
            alpha: 0.5
          },
          documents: new Set([documentId]),
          lastAccessed: Date.now(),
          strength: 1
        });
      } else {
        const node = newNodes.get(currentWord)!;
        node.count++;
        node.documents.add(documentId);
        node.lastAccessed = Date.now();
        node.strength = Math.min(node.strength + 0.1, MAX_MEMORY_STRENGTH);
        
        if (node.next.has(nextWord)) {
          const nextCount = node.next.get(nextWord)!;
          nextCount.count++;
        } else {
          node.next.set(nextWord, { word: nextWord, count: 1 });
        }
      }
    }
    
    setStats({
      totalWords: newNodes.size,
      activeMemories: Array.from(newNodes.values()).filter(n => n.strength >= 0.1).length
    });

    setMemory(prev => ({
      ...prev,
      nodes: newNodes,
      totalDocuments: prev.totalDocuments + 1
    }));
    
    saveMemory();
  }, [memory, saveMemory]);

  const generateResponse = useCallback((input: string) => {
    const words = input.toLowerCase().match(/\b\w+\b/g) || [];
    if (words.length === 0) return '';
    
    let currentWord = words[words.length - 1];
    const response = [...words];
    const maxLength = 20;
    
    while (response.length < maxLength) {
      const node = memory.nodes.get(currentWord);
      if (!node || node.next.size === 0) break;
      
      node.lastAccessed = Date.now();
      node.strength = Math.min(node.strength + 0.05, MAX_MEMORY_STRENGTH);
      
      const nextWords = Array.from(node.next.entries());
      nextWords.sort((a, b) => b[1].count - a[1].count);
      
      const totalStrength = nextWords.reduce((sum, [word]) => {
        const node = memory.nodes.get(word);
        return sum + (node?.strength || 0);
      }, 0);
      
      let random = Math.random() * totalStrength;
      let nextWord = nextWords[0][0];
      
      for (const [word] of nextWords) {
        const node = memory.nodes.get(word);
        if (!node) continue;
        random -= node.strength;
        if (random <= 0) {
          nextWord = word;
          break;
        }
      }
      
      response.push(nextWord);
      currentWord = nextWord;
      
      if (response.length >= 5 && Math.random() < 0.2) break;
    }
    
    saveMemory();
    return response.join(' ');
  }, [memory, saveMemory]);

  const updateHologram = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    if (!gl || !program) return;
    
    gl.useProgram(program);
    
    const timeLocation = gl.getUniformLocation(program, "u_time");
    const colorsLocation = gl.getUniformLocation(program, "u_wordColors");
    const positionsLocation = gl.getUniformLocation(program, "u_wordPositions");
    const strengthsLocation = gl.getUniformLocation(program, "u_wordStrengths");
    const countLocation = gl.getUniformLocation(program, "u_wordCount");
    
    gl.uniform1f(timeLocation, performance.now());
    
    const colors: number[] = [];
    const positions: number[] = [];
    const strengths: number[] = [];
    let index = 0;
    
    const sortedNodes = Array.from(memory.nodes.values())
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 128);
    
    for (const node of sortedNodes) {
      if (index >= 128) break;
      
      colors.push(
        node.color.hue / 360,
        node.color.saturation / 100,
        node.color.brightness / 100,
        node.color.alpha * (node.strength / MAX_MEMORY_STRENGTH)
      );
      
      const radius = 0.4 * (1 - node.strength / MAX_MEMORY_STRENGTH);
      const angle = (Math.PI * 2 * index) / Math.min(sortedNodes.length, 128);
      positions.push(
        0.5 + Math.cos(angle) * radius,
        0.5 + Math.sin(angle) * radius
      );
      
      strengths.push(node.strength / MAX_MEMORY_STRENGTH);
      
      index++;
    }
    
    gl.uniform4fv(colorsLocation, new Float32Array(colors));
    gl.uniform2fv(positionsLocation, new Float32Array(positions));
    gl.uniform1fv(strengthsLocation, new Float32Array(strengths));
    gl.uniform1f(countLocation, index);
    
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    rafRef.current = requestAnimationFrame(updateHologram);
  }, [memory]);

  useEffect(() => {
    updateHologram();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [updateHologram]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { alpha: true });
    if (!gl) return;
    
    glRef.current = gl;
    const program = createShaderProgram(gl);
    if (!program) return;
    
    programRef.current = program;
    const positions = new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      -1, 1, 0,
      1, 1, 0
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0,   0);

    const resizeCanvas = () => {
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
      
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  const createShaderProgram = (gl: WebGLRenderingContext) => {
    const vShader = gl.createShader(gl.VERTEX_SHADER)!;
    const fShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(vShader, vertexShader);
    gl.shaderSource(fShader, fragmentShader);
    gl.compileShader(vShader);
    gl.compileShader(fShader);

    const program = gl.createProgram()!;
    gl.attachShader(program, vShader);
    gl.attachShader(program, fShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Unable to initialize the shader program:', gl.getProgramInfoLog(program));
      return null;
    }

    return program;
  };

  const handleInput = () => {
    if (!inputText.trim()) return;
    processText(inputText, `input_${Date.now()}`);
    const response = generateResponse(inputText);
    setAnswer(response);
    setInputText('');
  };

  const handleQuestionSubmit = () => {
    if (!question.trim()) return;
    const answer = generateResponse(question);
    setAnswer(answer);
    setQuestion('');
  };

  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-6 h-6" />
            Holographic Neural Memory Processor
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Upload Document</Label>
            <Input
              type="file"
              accept=".txt"
              onChange={handleFileUpload}
              className="w-full"
            />
            {loading && <Progress value={progress} className="w-full" />}
          </div>

          <Separator />

          <div className="space-y-2">
            <Label>Input Text</Label>
            <div className="flex gap-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter your text..."
                className="flex-1"
              />
              <Button onClick={handleInput}>
                <MessageSquare className="w-4 h-4 mr-2" />
                Process
              </Button>
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label>Ask a Question</Label>
            <div className="flex gap-2">
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask about the processed text..."
                className="flex-1"
              />
              <Button onClick={handleQuestionSubmit}>
                <FileQuestion className="w-4 h-4 mr-2" />
                Ask
              </Button>
            </div>
          </div>

          {answer && (
            <Alert>
              <AlertDescription>{answer}</AlertDescription>
            </Alert>
          )}

          <Separator />

          <div className="space-y-2">
            <Label>Memory Statistics</Label>
            <div>Total Words: {stats.totalWords}</div>
            <div>Active Memories: {stats.activeMemories}</div>
            <div>Total Documents: {memory.totalDocuments}</div>
          </div>

          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="w-full h-96 border rounded-lg bg-black"
          />

          <div className="flex justify-between">
            <Button onClick={consolidateMemory}>
              <Database className="w-4 h-4 mr-2" />
              Consolidate Memory
            </Button>
            <Button onClick={saveMemory}>
              <Save className="w-4 h-4 mr-2" />
              Save Memory
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}