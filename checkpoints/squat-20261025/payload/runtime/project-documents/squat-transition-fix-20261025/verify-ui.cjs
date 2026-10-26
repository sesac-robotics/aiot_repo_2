const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync('output/live-webcam-20261023/index.html','utf8').split('<script>')[1].split('</script>')[0];
const nodes={};function node(){return {textContent:'',value:'',hidden:false,disabled:false,style:{},append(){},insertRow:node,insertCell:node,createTBody:node,addEventListener(){}};}
const document={getElementById(id){return nodes[id]??=node()},createElement:node,body:{classList:{add(){}}}};
nodes.exercise={...node(),value:'squats'};nodes.view={...node(),value:'front'};
let requests=[];const context=vm.createContext({document,URLSearchParams,location:{search:''},Image:class{},setTimeout(){},setInterval(){},fetch(url){return new Promise(resolve=>requests.push({url,resolve}))}});
vm.runInContext(source,context);
const run=s=>vm.runInContext(s,context);
const old={generation:3,exercise:'shoulder',view:'front',started_at:10,movement:{count:8},repetitions:[{number:8}],last_rep:{number:8},live_result:{old:true},elapsed_seconds:44.6};
context.old=old;
let checks=0;function check(expr){assert.ok(run(expr),expr);checks++}
check("sessionProjection(old,'squats','front',null).repetitions.length===0");
check("sessionProjection(old,'shoulder','side',null).movement.count===0");
check("sessionProjection(old,'shoulder','front',{generation:3}).last_rep===null");
check("sessionProjection({...old,start_reservation:{request_id:'a'}},'shoulder','front',null).elapsed_seconds===0");
check("sessionProjection(old,'shoulder','front',{generation:3}).live_result===null");
check("sessionProjection({...old,generation:4,collecting:true,repetitions:[]},'shoulder','front',{generation:3}).started_at===10");
check("sessionProjection({...old,generation:4},'shoulder','front',{generation:3}).movement.count===8");
check("sessionProjection({...old,start_reservation:{state:'cancelled'}},'shoulder','front',{generation:3}).repetitions.length===0");
check("old.movement.count===8 && old.repetitions.length===1");
(async()=>{
// An in-flight response must not repaint after selection/reset.
run('clearCurrentDisplay()');requests.shift().resolve({json:async()=>old});await new Promise(setImmediate);
assert.equal(nodes.count.textContent,'0회');checks++;
// Reload into a pending same-exercise reservation, then cancel: old results stay hidden.
run("sessionFence=null;el('exercise').value='shoulder'");let p=run('refresh()');requests.shift().resolve({json:async()=>({...old,start_reservation:{request_id:'pending',exercise:'shoulder',view:'front'}})});await p;
assert.equal(nodes.count.textContent,'0회');checks++;
// Heartbeat is fire-and-forget, resolve separately.
requests.shift().resolve({ok:true,json:async()=>({})});
p=run('refresh()');requests.shift().resolve({json:async()=>({...old,start_reservation:{state:'cancelled'}})});await p;
assert.equal(nodes.count.textContent,'0회');assert.equal(nodes.history.textContent,'');checks+=2;
// A new generation starts clean and observations dropping do not restore old data.
p=run('refresh()');requests.shift().resolve({json:async()=>({...old,generation:4,collecting:true,movement:{count:0,phase:'観測'},repetitions:[],last_rep:null})});await p;
assert.equal(nodes.count.textContent,'완료 검출 없음');checks++;
// Older generation after start is ignored.
p=run('refresh()');requests.shift().resolve({json:async()=>old});await p;assert.equal(nodes.count.textContent,'완료 검출 없음');checks++;
// First completed repetition of the new session is displayed normally.
p=run('refresh()');requests.shift().resolve({json:async()=>({...old,generation:4,collecting:true,movement:{count:1,phase:'ready'},repetitions:[{number:1}],last_rep:null})});await p;
assert.equal(nodes.count.textContent,'1회');checks++;
// A selection event cannot clear or switch a collecting session.
run("el('exercise').value='plank'");await run('selectionChanged()');assert.equal(nodes.exercise.value,'shoulder');assert.equal(nodes.count.textContent,'1회');checks+=2;
// Overlapping polling is suppressed.
p=run('refresh()');const pendingCount=requests.length;await run('refresh()');assert.equal(requests.length,pendingCount);checks++;
requests.shift().resolve({json:async()=>({...old,generation:4,collecting:false,repetitions:[],last_rep:null})});await p;
// Restart clears immediately, even before the arm response.
run('latestState={...old,generation:4,collecting:false};clearCurrentDisplay()');assert.equal(nodes.aiTotal.textContent,'— /100');assert.equal(nodes.items.textContent,'');checks+=2;
// A server restart has a new identity and may restart generation from zero.
p=run('refresh()');requests.shift().resolve({json:async()=>({...old,server_instance_id:'new-runtime',generation:0,started_at:null,movement:{count:0},repetitions:[],last_rep:null})});await p;
assert.equal(nodes.count.textContent,'0회');assert.equal(run('maxGeneration'),0);checks+=2;
p=run('refresh()');requests.shift().resolve({json:async()=>({...old,server_instance_id:'new-runtime',generation:1,collecting:true,movement:{count:1},repetitions:[{number:1}],last_rep:null})});await p;
assert.equal(nodes.count.textContent,'1회');checks++;
console.log(JSON.stringify({checks,passed:true,server_data_unchanged:old.movement.count===8}));
})().catch(e=>{console.error(e);process.exitCode=1});
