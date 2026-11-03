import {evaluate,interpolate,rated} from './model.js';
import {site} from './site-data.js';
export const D=site.diameter, R=D/2,H=site.hubHeight;
export const farmDefaults={layoutMode:'synthetic',ctMode:'curve',rows:6,cols:8,spacingX:5,spacingZ:7,staggered:false,k:.04,ct:.8,wakeEnabled:true};
export function layout(c=farmDefaults){if(c.layoutMode==='synthetic')return site.positions.map(p=>({...p}));return Array.from({length:c.rows*c.cols},(_,i)=>{const row=Math.floor(i/c.cols),col=i%c.cols;return {id:`WT-${String(i+1).padStart(2,'0')}`,x:(col-(c.cols-1)/2+(c.staggered&&row%2?.5:0))*c.spacingX*D,z:(row-(c.rows-1)/2)*c.spacingZ*D};});}
export function downwind(direction){const a=direction*Math.PI/180;return {x:-Math.sin(a),z:Math.cos(a)};}
// Exact circle intersection area, normalized by downstream rotor disk area.
export function overlap(radius,distance){const r=R,q=radius,d=Math.abs(distance);if(d>=q+r)return 0;if(d<=Math.abs(q-r))return Math.min(q*q,r*r)/(r*r);const clamp=v=>Math.max(-1,Math.min(1,v));const area=r*r*Math.acos(clamp((d*d+r*r-q*q)/(2*d*r)))+q*q*Math.acos(clamp((d*d+q*q-r*r)/(2*d*q)))-.5*Math.sqrt(Math.max(0,(-d+r+q)*(d+r-q)*(d-r+q)*(d+r+q)));return area/(Math.PI*r*r);}
export function wakeContribution(source,target,flow,c,disk=true){
 const dx=target.x-source.x,dz=target.z-source.z,x=dx*flow.x+dz*flow.z;
 if(x<=1e-6||source.ct<=0||!c.wakeEnabled)return {deficit:0,coverage:0,distance:x};
 const cross=Math.hypot(dx*flow.z-dz*flow.x,disk?0:(target.y??H)-H),radius=R+c.k*x;
 const coverage=disk?overlap(radius,cross):(cross<=radius?1:0);
 const deficit=source.wind*(1-Math.sqrt(1-source.ct))/(1+c.k*x/R)**2*coverage;
 return {deficit,coverage,distance:x};
}
export function solveFarm(e,c,positions,previous=[],dt=0){
 const flow=downwind(e.direction),solved=[],byId=new Map(previous.map(t=>[t.id,t]));
 const order=[...positions].sort((a,b)=>(a.x-b.x)*flow.x+(a.z-b.z)*flow.z);
 for(const p of order){let sum=0;const upstream=[];for(const s of solved){const v=wakeContribution(s,p,flow,c);sum+=v.deficit**2;if(v.deficit>1e-6)upstream.push({id:s.id,...v});}
  const wind=Math.max(0,e.wind-Math.sqrt(sum)),old=byId.get(p.id)?.state;
  const state=evaluate({...e,wind},old,dt),ct=state.power>0?(c.ctMode==='curve'?interpolate(site.ctCurve,wind):c.ct):0;
  solved.push({...p,wind,ct,deficit:e.wind-wind,upstream,state});
 }
 const turbines=positions.map(p=>solved.find(s=>s.id===p.id));
 const noWakePerTurbine=evaluate(e,undefined,0).power,baseline=noWakePerTurbine*positions.length;
 const power=turbines.reduce((sum,t)=>sum+t.state.power,0),loss=Math.abs(baseline-power)<1e-10?0:baseline-power;
 return {turbines,power,baseline,loss,lossPercent:baseline>0?100*loss/baseline:0,flow,operating:turbines.filter(t=>t.state.power>0).length,capacity:positions.length*rated};
}
// Point velocity at hub height, using the same solved upstream strengths as the rotor model.
export function pointWind(x,z,e,c,farm,y=H){let sum=0;for(const t of farm.turbines){sum+=wakeContribution(t,{x,z,y},farm.flow,c,false).deficit**2;}return Math.max(0,e.wind-Math.sqrt(sum));}
