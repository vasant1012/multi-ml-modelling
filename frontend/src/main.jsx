import React, {useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const ALL = ["sentiment", "rating", "emotion", "aspects", "summary"];
function App(){
  const [text,setText]=useState("Great location and friendly staff, but the room was noisy.");
  const [rating,setRating]=useState("");
  const [selected,setSelected]=useState(ALL);
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const toggle=(name)=>setSelected(v=>v.includes(name)?v.filter(x=>x!==name):[...v,name]);
  async function analyze(){
    setLoading(true); setError(""); setResult(null);
    try{
      const response=await fetch("http://localhost:8000/api/v1/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,rating:rating?Number(rating):null,components:selected})});
      if(!response.ok) throw new Error(await response.text());
      setResult((await response.json()).result);
    }catch(e){setError(e.message)}finally{setLoading(false)}
  }
  return <main><h1>Hotel Review Intelligence</h1><p>Analyze one review through selectable LangGraph components.</p>
    <label>Review<textarea rows="7" value={text} onChange={e=>setText(e.target.value)}/></label>
    <label>Original rating, optional<input type="number" min="1" max="5" value={rating} onChange={e=>setRating(e.target.value)}/></label>
    <section className="checks">{ALL.map(x=><label key={x}><input type="checkbox" checked={selected.includes(x)} onChange={()=>toggle(x)}/>{x}</label>)}</section>
    <button disabled={loading||!text.trim()||!selected.length} onClick={analyze}>{loading?"Analyzing...":"Analyze review"}</button>
    {error&&<pre className="error">{error}</pre>}
    {result&&<section className="result"><h2>Result</h2><pre>{JSON.stringify(result,null,2)}</pre></section>}
  </main>
}
createRoot(document.getElementById("root")).render(<App/>);
