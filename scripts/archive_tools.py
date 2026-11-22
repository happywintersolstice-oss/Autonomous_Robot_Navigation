"""Read-only historical ZIP export and integrity validation (stdlib only)."""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_member(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe ZIP member: '+name)
    return p.as_posix()

def load_manifest(root):
    return json.loads((Path(root)/'archive/manifests/pre-migration.json').read_text(encoding='utf-8'))

def verify_sources(root,manifest=None):
    root=Path(root).resolve();m=manifest or load_manifest(root);cache={};errors=[]
    for row in m['files']:
        p=(root/row['archive_path']).resolve()
        if not p.is_relative_to(root):errors.append(row['archive_path']+': escapes repository');continue
        if not p.is_file():errors.append(row['archive_path']+': missing');continue
        if str(p) not in cache:cache[str(p)]=(p.stat().st_size,file_hash(p))
        if cache[str(p)]!=(row['bytes'],row['sha256']):errors.append(row['archive_path']+': content mismatch')
        if row.get('lfs_missing'):errors.append(row['original_path']+': missing LFS payload '+row['lfs_oid'])
    if errors:raise ValueError('Archive integrity failed:\n'+'\n'.join(errors)+'\nRecover missing LFS objects with git lfs fetch --all and git lfs pull; do not invent payloads.')
    return {'status':'passed','file_versions':len(m['files']),'unique_backing_files':len(cache),'bytes':sum(r['bytes'] for r in m['files']),'missing_files':0,'missing_lfs':0}

def enumerate_exports(root,manifest=None):
    m=manifest or load_manifest(root)
    result={}
    for collection in ('classic','partial'):
        members={}
        for row in m['files']:
            if row['collection'] not in (collection,'both'):continue
            members[row['zip_path']]=row
            # Restore the complete original snapshot hierarchy, including aliases
            # whose bytes were deduplicated into an embedded source snapshot.
            for origin in row.get('origins',[]):
                if origin['source_version']==m.get('snapshot_parent') and origin.get('representation')=='on-disk':
                    member='original/'+origin['path']
                    members.setdefault(member,{**row,'zip_path':member,'original_path':origin['path']})
        result[collection]=list(members.values())
    return result

def export(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);m=load_manifest(root)
    # Each file is hashed as it streams into the ZIP; no duplicate pre-export pass.
    inventory={'schema':'robot-history-export-v1','source_manifest_sha256':file_hash(root/'archive/manifests/pre-migration.json'),'archives':{},'files':[]}
    for collection,rows in enumerate_exports(root,m).items():
        name='legacy-classic-history.zip' if collection=='classic' else 'partial-navigation-research-history.zip';dest=output/name
        if dest.exists():raise FileExistsError('Choose a fresh output folder: '+str(dest))
        with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for row in rows:
                if row.get('lfs_missing'):raise ValueError('Missing LFS payload: '+row['original_path']+'; git lfs fetch --all / git lfs pull')
                source=(root/row['archive_path']).resolve();member=safe_member(row['zip_path']);h=hashlib.sha256();size=0
                if not source.is_relative_to(root):raise ValueError('Archive backing path escapes repository')
                info=zipfile.ZipInfo(member,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100444<<16
                with source.open('rb') as src,z.open(info,'w',force_zip64=True) as dst:
                    for block in iter(lambda:src.read(1024*1024),b''):h.update(block);size+=len(block);dst.write(block)
                if h.hexdigest()!=row['sha256'] or size!=row['bytes']:raise ValueError('Source changed while exporting: '+str(source))
                inventory['files'].append({'archive':name,'path':member,'bytes':size,'sha256':h.hexdigest(),'category':row['category'],
                    'source_version':row['source_version'],'original_path':row['original_path'],'export_status':'included'})
            info=zipfile.ZipInfo('ARCHIVE-ORIGINS.json',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100444<<16
            z.writestr(info,json.dumps({'files':rows,'notes':m['notes'],'source_refs':m.get('source_refs',[]),'source_versions':m.get('source_versions',{})},indent=2))
        inventory['archives'][name]={'sha256':file_hash(dest),'files':len(rows),'bytes':dest.stat().st_size}
    (output/'archive-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    return inventory

def verify_zip(output):
    output=Path(output);i=json.loads((output/'archive-inventory.json').read_text(encoding='utf-8'));count=0
    expected={name:{r['path']:r for r in i['files'] if r['archive']==name} for name in i['archives']}
    for name,meta in i['archives'].items():
        path=output/safe_member(name)
        if file_hash(path)!=meta['sha256']:raise ValueError('ZIP checksum mismatch: '+name)
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(expected[name])|{'ARCHIVE-ORIGINS.json'}:raise ValueError('ZIP entry coverage mismatch')
            for member,row in expected[name].items():
                safe_member(member);h=hashlib.sha256();n=0
                with z.open(member) as f:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
                if n!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('ZIP content mismatch: '+member)
                count+=1
    return {'status':'passed','zip_files':len(i['archives']),'verified_members':count}
