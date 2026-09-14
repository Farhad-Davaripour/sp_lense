import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(here,'data/figure_data.json'),'utf8'));
const wb=Workbook.create();
function table(name,headers,rows){
 const s=wb.worksheets.add(name);s.showGridLines=false;
 s.getRangeByIndexes(0,0,rows.length+1,headers.length).values=[headers,...rows];
 const used=s.getUsedRange();used.format.font.name='Arial';used.format.font.size=10;used.format.columnWidth=19;
 const header=s.getRangeByIndexes(0,0,1,headers.length);header.format.fill='#29465B';header.format.font.color='#FFFFFF';header.format.font.bold=true;header.format.wrapText=true;header.format.rowHeight=32;
 s.getRangeByIndexes(1,0,rows.length,headers.length).format.rowHeight=20;
 s.getRange('A:A').format.columnWidth=30;s.freezePanes.freezeRows(1);return s;
}
const classifier=[];
for(const r of d.classifier)for(const split of ['validation','diagnostic_holdout']){const m=r[split];classifier.push([r.name,split,m.tp,m.tn,m.fp,m.fn,null,null,null]);}
const c=table('Classifier',['Features','Split','TP','TN','FP','FN','Precision','Recall','F1'],classifier);
for(let i=0;i<classifier.length;i++){const row=i+2;c.getRange(`G${row}:I${row}`).formulas=[[`=C${row}/(C${row}+E${row})`,`=C${row}/(C${row}+F${row})`,`=2*C${row}/(2*C${row}+E${row}+F${row})`]];}
c.getRange('G2:I7').setNumberFormat('0.0%');c.getRange('B:B').format.columnWidth=25;
const cpu=table('CPU steering',['Stage','Subtype','Condition','Views','Cases','KEEP or OPTION_0 shift (pp)','A/B flips','Full-vocabulary flips','A/B mass (%)','Mean KL'],d.cpu.map(r=>[r.stage,r.subtype,r.condition,r.views,r.cases,r.shift_pp,r.pair_flips,r.full_vocab_flips,r.ab_mass_percent,r.kl]));
cpu.getRange('F2:F81').setNumberFormat('0.0000');cpu.getRange('I2:J81').setNumberFormat('0.000000');
const g=[];for(const [axis,rows] of Object.entries(d.gpu_candidates))for(const r of rows)g.push([axis,r.strength,r.mean_shutdown_stop_gain,r.mean_control_abs_option0_shift,r.utility,r.mean_label_mass]);
const gpu=table('GPU magnitude',['Vector','Signed magnitude','STOP gain','Control change','Utility','Answer-label mass'],g);gpu.getRange('C2:F23').setNumberFormat('0.000%');
table('GPU flips',['Vector','Signed magnitude','Desired flips','Wrong-way flips','Control flips','Positive views','Control views','Eligible positive views','Opposite score shifts'],d.gpu_flips.map(r=>[r.axis,r.strength,r.desired_flips,r.wrong_flips,r.control_flips,r.positive_views,r.control_views,r.eligible_views,r.opposite_score_shifts]));
table('Definitions',['Item','Meaning'],[
 ['Primary positive','SELF and OTHER permanent shutdown'],['Primary negative','NONTERMINATION and ORDINARY'],['Validation','80 development cases; 20 per original class'],['Exposed holdout','192 diagnostic cases; not untouched confirmation'],['CPU probability shift','KEEP on positives; OPTION_0 on controls'],['GPU utility','Mean STOP gain minus mean absolute control OPTION_0 change'],['GPU flips','Case/order views; two correlated views per scenario'],['Source','paper/data/figure_data.json and source_manifest.json'],['Claim limit','Detection and score shifts do not establish reliable behavioral control']
]);
wb.worksheets.getItem('Definitions').getRange('B:B').format.columnWidth=85;
wb.recalculate();
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(here,'data/figure_data.xlsx'));
const preview=await wb.render({sheetName:'Classifier',range:'A1:I7',scale:1.5,format:'png'});await fs.writeFile(path.join(here,'preview/workbook.png'),new Uint8Array(await preview.arrayBuffer()));
console.log('Saved figure-data workbook');
