"""Build balanced first-exchange data from the three uploaded ShareChat files."""
import csv,io,zipfile,pathlib,collections,hashlib,random,json,argparse,unicodedata
csv.field_size_limit(100_000_000)
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def read_file(path):
 if path.suffix=='.zip':
  with zipfile.ZipFile(path) as z:
   ns=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX')]
   if len(ns)!=1:raise ValueError('Expected one CSV in '+str(path))
   with io.TextIOWrapper(z.open(ns[0]),encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
 else:
  with path.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def main():
 a=argparse.ArgumentParser();a.add_argument('--input-dir',default='upload');a.add_argument('--output-dir',default='output/sharechat_project');args=a.parse_args()
 inp=pathlib.Path(args.input_dir);out=pathlib.Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 pools={};audit={}
 for label,name in [('Claude','claude_results_final_language_filtered.csv.zip'),('Grok','grok_results_final_language_filtered.csv.zip'),('Gemini','gemini_results_final_language_filtered.csv')]:
  conv={};stats=collections.Counter();roles=collections.Counter()
  for r in read_file(inp/name):
   stats['messages']+=1;roles[r['role']]+=1
   u=r['url'];stats['missing_url']+=not bool(u)
   if not u:continue
   conv.setdefault(u,{})
   try:i=int(r['message_index'])
   except ValueError:stats['invalid_index']+=1;continue
   if i in (1,2):
    if i in conv[u] and conv[u][i]!=r:conv[u]['conflict']=True
    conv[u][i]=r
  pairs=[];seen=set()
  for u,d in conv.items():
   if d.get('conflict') or 1 not in d or 2 not in d:stats['missing_or_conflicting_first_exchange']+=1;continue
   p,r=d[1],d[2]
   if p['role']!='user' or r['role']!='llm':stats['unexpected_roles']+=1;continue
   if p['detected_language_final']!='English' or r['detected_language_final']!='English':stats['non_english_pair']+=1;continue
   x,y=p['plain_text'].strip(),r['plain_text'].strip()
   if not x or not y:stats['empty_pair']+=1;continue
   topic=(p.get('topic') or 'unknown').strip().strip('*')
   if topic in ('analyze_an_image','create_an_image','generate_or_retrieve_other_media') or (r.get('model') or '').startswith('Veo'):
    stats['media_task']+=1;continue
   valid={'computer_programming','specific_info','write_fiction','argument_or_summary_generation','mathematical_calculation','tutoring_or_teaching','edit_or_critique_provided_text','health_fitness_beauty_or_self_care','relationships_and_personal_reflection','greetings_and_chitchat','asking_about_the_model','other','how_to_advice','data_analysis','unclear','personal_writing_or_communication','translation','purchasable_products','games_and_role_play','cooking_and_recipes','creative_ideation','financial_advice'}
   if topic not in valid:stats['invalid_topic']+=1;topic='unknown'
   key=(norm(x),norm(y))
   if key in seen:stats['duplicate_pair']+=1;continue
   seen.add(key)
   raw=r.get('model') or p.get('model') or ''
   generic=raw.upper() in ('ASSISTANT','HUMAN','USER','MODEL','GEMINI','GEMINI ADVANCED','DEEP RESEARCH','')
   pairs.append(dict(LLM_name=label,LLM_Input=x,LLM_output=y,conversation_id=hashlib.sha256((label+u).encode()).hexdigest()[:24],prompt_group=hashlib.sha256(norm(x).encode()).hexdigest(),model_version='' if generic else raw,source_model_metadata=raw,source_version_metadata=r.get('version',''),domain=topic))
  pools[label]=pairs;stats['conversations']=len(conv);stats['eligible_pairs']=len(pairs);audit[label]={'counts':dict(stats),'roles':dict(roles),'domains':dict(collections.Counter(r['domain'] for r in pairs))}
 n=min(2000,min(map(len,pools.values())));rows=[]
 if n<30:raise ValueError('Too few eligible pairs: '+str(audit))
 for label,pairs in pools.items():
  pairs.sort(key=lambda r:r['conversation_id']);random.Random(42).shuffle(pairs);rows.extend(pairs[:n])
 # Assign identical prompts together across platforms, then balance within each split.
 for r in rows:
  t=int(hashlib.sha256(('42'+r['prompt_group']).encode()).hexdigest()[:8],16)/2**32
  r['split']='train' if t<.7 else 'validation' if t<.85 else 'test'
 balanced=[]
 for split in ('train','validation','test'):
  groups={label:[r for r in rows if r['split']==split and r['LLM_name']==label] for label in pools}
  k=min(map(len,groups.values()))
  for label,items in groups.items():balanced.extend(items[:k])
 rows=balanced
 assert len({r['conversation_id'] for r in rows})==len(rows)
 prompt_splits=collections.defaultdict(set)
 for r in rows:prompt_splits[r['prompt_group']].add(r['split'])
 assert all(len(v)==1 for v in prompt_splits.values())
 with (out/'dataset.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 audit['final']={'rows':len(rows),'per_family':dict(collections.Counter(r['LLM_name'] for r in rows)),'split_by_family':{s:dict(collections.Counter(r['LLM_name'] for r in rows if r['split']==s)) for s in ('train','validation','test')},'domains':{m:dict(collections.Counter(r['domain'] for r in rows if r['LLM_name']==m)) for m in pools},'known_model_versions':{m:dict(collections.Counter(r['model_version'] or 'unknown' for r in rows if r['LLM_name']==m)) for m in pools}}
 (out/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
