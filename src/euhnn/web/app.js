"use strict";
/** Local, source-grounded workbench. Untrusted document text never becomes HTML. */
const $ = id => document.getElementById(id);
let token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
if (location.hash) history.replaceState(null, "", location.pathname);
let working = false;
let lastQuery = "";
let documentOffset = 0;
let sceneData = null;
let yaw = -0.52, pitch = 0.20, zoom = 1;
const pageSize = 50;

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

async function api(path, {method = "GET", json, body} = {}) {
  const headers = {Authorization: `Bearer ${token}`};
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch(path, {method, headers, body, signal: controller.signal, credentials: "omit", cache: "no-store"});
    if (!response.ok) {
      let message = `Request failed (${response.status}).`;
      try {
        const error = await response.json();
        message = error.message || error.error || (response.status === 422 ? "Check the request fields and limits." : message);
      } catch { /* A bounded generic error is preferable to displaying arbitrary HTML. */ }
      if (response.status === 401) $("connection-panel").hidden = false;
      throw new Error(message);
    }
    return response;
  } finally { clearTimeout(timer); }
}

async function operation(message, task) {
  if (working) return;
  working = true;
  document.querySelectorAll("button,input[type=file]").forEach(node => node.disabled = true);
  notice(message);
  try { await task(); }
  catch (error) { notice(error.name === "AbortError" ? "The operation exceeded the browser deadline. Refresh the index status before retrying." : error.message, true); }
  finally {
    working = false;
    document.querySelectorAll("button,input[type=file]").forEach(node => node.disabled = false);
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function refresh() {
  const [statusResponse, docsResponse] = await Promise.all([
    api("/api/status"), api(`/api/documents?limit=${pageSize}&offset=${documentOffset}`)]);
  const status = await statusResponse.json();
  const {documents} = await docsResponse.json();
  $("runtime").textContent = status.backend === "cuda" ? `CUDA | ${status.device}` : "CPU | NumPy reference";
  $("runtime").title = status.fallback_reason || "All document processing runs locally.";
  $("document-count").textContent = status.documents;
  $("passage-count").textContent = status.chunks.toLocaleString();
  $("teaching-count").textContent = status.teaching;
  $("previous-docs").hidden = documentOffset === 0;
  $("next-docs").hidden = documentOffset + pageSize >= status.documents;
  const list = $("documents"); list.replaceChildren();
  for (const doc of documents) {
    const row = element("div", undefined, "document");
    const description = element("div");
    description.append(element("strong", doc.title), element("p", `${doc.chunks} passages | ${doc.source}`));
    const remove = element("button", "Remove", "quiet");
    remove.type = "button";
    remove.addEventListener("click", () => {
      if (!confirm(`Remove "${doc.title}" and its learned associations from this local index?`)) return;
      operation("Removing document...", async () => {
        await api(`/api/documents/${encodeURIComponent(doc.document_id)}`, {method:"DELETE"});
        documentOffset = 0; await refresh();
        $("results").replaceChildren(element("p", "Document removed. Search again for current results.", "muted"));
        notice("Document and associated index entries removed.");
      });
    });
    row.append(description, remove); list.append(row);
  }
  if (!documents.length) list.append(element("p", "No documents on this page.", "muted small"));
}

async function updateScene(query) {
  sceneData = await (await api("/api/scene", {method:"POST", json:{query}})).json();
  drawScene();
}

function renderResults(result) {
  const root = $("results"); root.replaceChildren();
  $("result-count").textContent = `${result.hits.length} matches | ${result.elapsed_ms.toFixed(1)} ms`;
  if (!result.hits.length) {
    const empty = element("div", undefined, "empty");
    empty.append(element("h3", "No matching source passage."), element("p", "Try different terms or load a relevant document. No answer has been invented."));
    root.append(empty); return;
  }
  result.hits.forEach((hit, i) => {
    const card = element("article", undefined, "result");
    card.dataset.chunkId = hit.chunk_id;
    card.append(element("h3", `${i+1}. ${hit.title}`), element("div", hit.citation, "citation"),
                element("pre", hit.text), element("div", `Ranking ${hit.score.toFixed(4)} | optical ${hit.optical_score.toFixed(4)} | learned ${hit.learned_score.toFixed(4)}. Scores are not probabilities.`, "scores"));
    const actions = element("div", undefined, "actions");
    const copy = element("button", "Copy with citation", "quiet");
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(`${hit.citation}\n\n${hit.text}`); notice("Passage and citation copied."); }
      catch { notice("Clipboard access was not granted. Select the passage to copy it manually.", true); }
    });
    const teach = element("button", "Teach this association", "quiet");
    teach.addEventListener("click", () => {
      const query = prompt("Enter a query that should retrieve this passage. This trains a supervised readout, not a language model.", lastQuery);
      if (!query || !query.trim()) return;
      operation("Learning the selected association...", async () => {
        const result = await (await api("/api/learn", {method:"POST", json:{query,chunk_id:hit.chunk_id}})).json();
        await refresh(); notice(`Association learned and stored. Training affinity: ${result.training_score.toFixed(4)}.`);
      });
    });
    actions.append(copy, teach); card.append(actions); root.append(card);
  });
}

async function search() {
  const query = $("query").value.trim();
  if (!query) { notice("Enter a search term or a question."); return; }
  lastQuery = query;
  const payload = {query, top_k:Number($("top-k").value), mode:$("mode").value, phrase:$("phrase").checked};
  const result = await (await api("/api/search", {method:"POST", json:payload})).json();
  renderResults(result);
  await updateScene(query);
  notice("Showing retrieved quotations. Source page and line references refer to extracted document text.");
}

function project(point, width, height) {
  const x=point[0], y=point[1], z=point[2]-3;
  const u=Math.cos(yaw)*x+Math.sin(yaw)*z;
  const z1=-Math.sin(yaw)*x+Math.cos(yaw)*z;
  const v=Math.cos(pitch)*y-Math.sin(pitch)*z1;
  const depth=Math.sin(pitch)*y+Math.cos(pitch)*z1;
  const scale=Math.min(width,height)*0.50*zoom/(8+depth);
  return [width*.50+u*scale*3, height*.5-v*scale*3, scale, depth];
}

function drawScene() {
  const canvas=$("scene"), context=canvas.getContext("2d");
  const ratio=Math.min(devicePixelRatio||1,2), width=canvas.clientWidth, height=canvas.clientHeight;
  if (!width || !height) return;
  canvas.width=Math.round(width*ratio); canvas.height=Math.round(height*ratio);
  context.setTransform(ratio,0,0,ratio,0,0); context.clearRect(0,0,width,height);
  if (!sceneData) {
    context.fillStyle="#97afcc"; context.font="13px system-ui";
    context.fillText("Connect to compute the optical scene.",24,height/2); return;
  }
  const sources=sceneData.sources.map(point=>project(point,width,height));
  const detectors=sceneData.detectors.map(point=>project(point,width,height));
  context.lineWidth=.6;
  for (const [s,d] of sceneData.rays) {
    const rgb=sceneData.rgb[d];
    context.strokeStyle=`rgba(${70+Math.round(rgb[0]*170)},${85+Math.round(rgb[1]*170)},${95+Math.round(rgb[2]*160)},0.14)`;
    context.beginPath();context.moveTo(sources[s][0],sources[s][1]);context.lineTo(detectors[d][0],detectors[d][1]);context.stroke();
  }
  for (const sphere of sceneData.spheres) {
    const [x,y,scale]=project(sphere,width,height);
    context.beginPath();context.arc(x,y,Math.max(2,sphere[3]*scale*3),0,Math.PI*2);
    context.fillStyle="#3f597522";context.fill();context.strokeStyle="#577f9b66";context.lineWidth=1;context.stroke();
  }
  sources.forEach(([x,y],i)=>{
    const strength=Math.min(1,sceneData.source_rgb[i].reduce((a,b)=>a+b,0)*20);
    context.beginPath();context.arc(x,y,1.4+strength*2,0,Math.PI*2);
    context.fillStyle=`rgba(133,225,206,${.2+strength*.8})`;context.fill();
  });
  detectors.forEach(([x,y],i)=>{
    const rgb=sceneData.rgb[i];const maximum=Math.max(...rgb);
    const values=rgb.map(value=>Math.round(25+230*Math.sqrt(value)));
    context.fillStyle=`rgb(${values.join(",")})`;context.beginPath();context.arc(x,y,2+maximum*3,0,Math.PI*2);context.fill();
  });
  context.fillStyle="#abc0d8";context.font="10px system-ui";
  context.fillText(`${sources.length} sources / ${sceneData.spheres.length} phase inclusions / ${detectors.length} RGB detectors`,12,height-12);
}

function drawSpectrum(result) {
  const canvas=$("spectrum"), context=canvas.getContext("2d");
  canvas.hidden=false;canvas.width=result.width;canvas.height=result.height;
  const image=context.createImageData(result.width,result.height);
  result.rgb.forEach((row,y)=>row.forEach((rgb,x)=>{
    const i=(y*result.width+x)*4;
    image.data[i]=Math.round(rgb[0]*255);image.data[i+1]=Math.round(rgb[1]*255);image.data[i+2]=Math.round(rgb[2]*255);image.data[i+3]=255;
  }));
  context.putImageData(image,0,0);
}

$("search-form").addEventListener("submit",event=>{event.preventDefault();operation("Retrieving original passages...",search);});
$("mode").addEventListener("change",()=>{const optical=$("mode").value==="optical";$("phrase").disabled=optical;if(optical)$("phrase").checked=false;});
$("connect-toggle").addEventListener("click",()=>{$("connection-panel").hidden=!$("connection-panel").hidden;});
$("connect").addEventListener("click",()=>operation("Connecting to local memory...",async()=>{
  token=$("connection-token").value.trim();$("connection-token").value="";
  await refresh();await updateScene("holographic optical memory");$("connection-panel").hidden=true;notice("Connected to the local index.");
}));
$("refresh").addEventListener("click",()=>operation("Refreshing library...",async()=>{await refresh();notice("Library refreshed.");}));
$("load-demo").addEventListener("click",()=>operation("Indexing bundled examples...",async()=>{
  await api("/api/demo",{method:"POST"});await refresh();$("query").value="blue wavelength";await search();
}));
$("add-text").addEventListener("click",()=>operation("Indexing source text...",async()=>{
  const result=await (await api("/api/text",{method:"POST",json:{source:$("source-name").value,text:$("source-text").value}})).json();
  await refresh();notice(`${result.unchanged?"Already indexed":"Indexed"}: ${result.chunks} source passages.`);
}));
$("file-input").addEventListener("change",event=>{
  const files=Array.from(event.target.files||[]);event.target.value="";
  operation("Indexing selected documents...",async()=>{
    for(let i=0;i<files.length;i++){
      const file=files[i];if(file.size>64*1024*1024)throw new Error(`${file.name} exceeds 64 MiB.`);
      notice(`Indexing ${i+1}/${files.length}: ${file.name}`);
      await api(`/api/upload/${encodeURIComponent(file.name)}`,{method:"PUT",body:file});
    }
    await refresh();notice(`Indexed ${files.length} selected documents. Existing sources were updated atomically.`);
  });
});
$("export-memory").addEventListener("click",()=>operation("Encoding exact source memory into RGB phase channels...",async()=>{
  const blob=await (await api("/api/memory/export")).blob();const url=URL.createObjectURL(blob);
  const link=element("a");link.href=url;link.download="euhnn-memory.holo";link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
  notice("Holographic memory exported. The file contains source text; it is not encrypted.");
}));
$("memory-input").addEventListener("change",event=>{
  const file=event.target.files?.[0];event.target.value="";if(!file)return;
  if(!confirm("Import this memory? Matching source names will be replaced in a single transaction. Unrelated documents remain."))return;
  operation("Decoding and verifying holographic memory...",async()=>{
    if(file.size>272*1024*1024)throw new Error("The hologram exceeds 272 MiB.");
    const result=await (await api("/api/memory/import",{method:"PUT",body:file})).json();await refresh();
    notice(`Imported ${result.documents} documents and ${result.chunks} exact passages.`);
  });
});
$("verify").addEventListener("click",()=>operation("Verifying index and source checksums...",async()=>{
  const result=await (await api("/api/verify")).json();notice(`Integrity verified: ${result.verified_passages} passages. These are corruption checks, not digital signatures.`);
}));
$("show-spectrum").addEventListener("click",()=>operation("Computing the stored memory spectrum...",async()=>{
  const result=await (await api("/api/memory/preview")).json();drawSpectrum(result);
  notice(`Actual RGB Fourier amplitudes: ${result.stored_side} x ${result.stored_side} per channel; ${result.raw_bytes.toLocaleString()} source bytes.`);
}));
$("previous-docs").addEventListener("click",()=>operation("Loading documents...",async()=>{documentOffset=Math.max(0,documentOffset-pageSize);await refresh();notice("Library page loaded.");}));
$("next-docs").addEventListener("click",()=>operation("Loading documents...",async()=>{documentOffset+=pageSize;await refresh();notice("Library page loaded.");}));
let drag=null;
$("scene").addEventListener("pointerdown",event=>{drag=[event.clientX,event.clientY];$("scene").setPointerCapture(event.pointerId);});
$("scene").addEventListener("pointermove",event=>{if(!drag)return;yaw+=(event.clientX-drag[0])*.008;pitch=Math.max(-1,Math.min(1,pitch+(event.clientY-drag[1])*.008));drag=[event.clientX,event.clientY];drawScene();});
$("scene").addEventListener("pointerup",()=>{drag=null;});
$("scene").addEventListener("pointercancel",()=>{drag=null;});
$("scene").addEventListener("wheel",event=>{event.preventDefault();zoom=Math.max(.5,Math.min(2,zoom*Math.exp(-event.deltaY*.001)));drawScene();},{passive:false});
new ResizeObserver(drawScene).observe($("scene"));
if(token)operation("Opening local memory...",async()=>{await refresh();await updateScene("holographic optical memory");notice("Ready. Add documents or load the bundled examples.");});
else $("connection-panel").hidden=false;
