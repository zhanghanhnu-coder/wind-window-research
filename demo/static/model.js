import {site} from './site-data.js';
export const rated=site.ratedMW;
export function interpolate(table,v){if(v<table[0][0]||v>table.at(-1)[0])return 0;for(let i=1;i<table.length;i++){if(v<=table[i][0]){const [x0,y0]=table[i-1],[x1,y1]=table[i];return y0+(y1-y0)*(v-x0)/(x1-x0);}}return table.at(-1)[1];}
// Reduced-order educational model; not a calibrated plant controller.
export const defaults={wind:9.5,direction:35,temp:22,wave:1.4,period:7,waveDirection:65,turbulence:10};
export function evaluate(e,previous={heat:45},dt=1){
 const rho=101325/(287.05*(e.temp+273.15));
 const available=interpolate(site.powerCurveKW,e.wind)/1000;
 const derate=Math.max(.4,1-Math.max(0,e.temp-35)*.03);
 const stopped=e.wind<3||e.wind>=25;
 const power=stopped?0:Math.min(rated,available)*derate;
 const target=e.temp+12+48*(power/rated)**2;
 const thermalDt=dt/(e.thermalTimeScale??1);
 const heat=previous.heat+(target-previous.heat)*(1-Math.exp(-thermalDt/45));
 const load=power/rated;
 const thermal=(key,offset,rise,tau)=>{const old=previous[key]??(e.temp+offset);return old+(e.temp+offset+rise*load*load-old)*(1-Math.exp(-thermalDt/tau));};
 const gearboxOilTemp=thermal('gearboxOilTemp',8,35,80);
 const gearboxBearingTemp=thermal('gearboxBearingTemp',10,42,65);
 const generatorBearingTemp=thermal('generatorBearingTemp',8,32,60);
 const cabinetTemp=thermal('cabinetTemp',5,25,50);
 const terminalTemp=thermal('terminalTemp',5,38,40);
 const rpm=stopped?0:Math.min(12.1,6.9+(e.wind-3)*.62);
 const status=e.wind>=25?'大风切出':e.wind<3?'低风待机':derate<1?'高温降额':power>=rated*.99?'额定运行':'并网发电';
 return {sessionEnergyMWh:(previous.sessionEnergyMWh??0)+power*dt/3600,sessionRunHours:(previous.sessionRunHours??0)+(power>0?dt/3600:0),power,rho,rpm,heat,gearboxOilTemp,gearboxBearingTemp,generatorBearingTemp,cabinetTemp,terminalTemp,equivalentTorque:rpm>0?power*1000/(rpm*2*Math.PI/60):0,current:power*1e6/(Math.sqrt(3)*690*.98),voltage:690,pf:.98,frequency:50,reactive:power*Math.tan(Math.acos(.98)),pitch:stopped?90:Math.max(0,(e.wind-11.4)*2.4),status,access:e.wind<=12&&e.wave<=1.5,load:Math.min(100,(e.wind/25)**2*55+(e.wave/6)**2*45)};
}
