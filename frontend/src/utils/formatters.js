export const percent=v=>`${Math.round((Number(v)||0)*100)}%`;export const titleCase=s=>(s||"unknown").replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());
