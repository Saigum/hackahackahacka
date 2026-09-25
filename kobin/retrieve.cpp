#include <bits/stdc++.h>
#include <omp.h>
#include <rapidfuzz/fuzz.hpp>
using namespace std;
struct Rec {string id,n,a,c; int nn=0,na=0; vector<int> nt,at;};
struct Term {vector<int> p; float w=0;};
vector<string> split(const string&s,char d=' ') {vector<string> v; string t;istringstream f(s);while(getline(f,t,d))v.push_back(t);return v;}
vector<Rec> load(string path){ifstream f(path);string l;vector<Rec> v;while(getline(f,l)){auto s=split(l,'\t');if(s.size()<6)continue;Rec r;r.id=s[0];r.n=s[1];r.a=s[2];r.c=s[3];r.nn=stoi(s[4]);r.na=stoi(s[5]);v.push_back(move(r));}return v;}
string compact(string s){s.erase(remove(s.begin(),s.end(),' '),s.end());return s;}
string consonants(string s){string r;for(char c:s)if(c!='a'&&c!='e'&&c!='i'&&c!='o'&&c!='u'&&c!=' ')if(r.empty()||r.back()!=c)r+=c;return r;}
string acronym(const string&s){string r;for(auto&w:split(s))if(!w.empty())r+=w[0];return r;}
vector<string> nums(const string&s){vector<string>r;string t;for(char c:s){if(isdigit(c))t+=c;else if(!t.empty()){r.push_back(t);t.clear();}}if(!t.empty())r.push_back(t);return r;}
void fuzzy(vector<float>&f,const string&a,const string&b){f.push_back(rapidfuzz::fuzz::ratio(a,b)/100);f.push_back(rapidfuzz::fuzz::token_sort_ratio(a,b)/100);f.push_back(rapidfuzz::fuzz::token_set_ratio(a,b)/100);f.push_back(rapidfuzz::fuzz::partial_ratio(a,b)/100);f.push_back(rapidfuzz::fuzz::ratio(compact(a),compact(b))/100);f.push_back(a==b&&!a.empty());f.push_back(float(min(a.size(),b.size()))/max(size_t(1),max(a.size(),b.size())));}
void overlap(vector<float>&f,const vector<int>&a,const vector<int>&b,const vector<Term>&terms){float wa=0,wb=0,wi=0;int common=0;for(int x:a)wa+=terms[x].w;for(int x:b)wb+=terms[x].w;for(int x:a)if(find(b.begin(),b.end(),x)!=b.end()){wi+=terms[x].w;common++;}f.push_back(common);f.push_back(float(common)/max(size_t(1),a.size()));f.push_back(float(common)/max(size_t(1),b.size()));f.push_back(wi/max(0.01f,wa));f.push_back(wi/max(0.01f,wb));f.push_back(wi);}
vector<float> features(const Rec&q,const Rec&t,const vector<Term>&terms,float sn,float sa,int rank){vector<float>f;fuzzy(f,q.n,t.n);fuzzy(f,q.a,t.a);overlap(f,q.nt,t.nt,terms);overlap(f,q.at,t.at,terms);f.push_back(rapidfuzz::fuzz::ratio(consonants(q.n),consonants(t.n))/100);f.push_back(rapidfuzz::fuzz::ratio(acronym(q.n),acronym(t.n))/100);auto a=nums(q.a),b=nums(t.a);int cnt=0;for(auto&x:a)if(find(b.begin(),b.end(),x)!=b.end())cnt++;f.push_back(cnt);f.push_back(a.size());f.push_back(b.size());f.push_back(!a.empty()&&!b.empty()&&a[0]==b[0]);f.push_back(float(cnt)/max(size_t(1),a.size()));f.push_back(float(cnt)/max(size_t(1),b.size()));f.push_back(t.n.size());f.push_back(t.a.size());f.push_back(q.n.size());f.push_back(q.a.size());f.push_back(t.nn);f.push_back(t.na);f.push_back(t.id[1]=='3');f.push_back(sn);f.push_back(sa);f.push_back(rank);return f;}
int main(int argc,char**argv){if(argc<6){cerr<<"targets2 targets3 queries output_prefix threads [K=50]\n";return 1;}int threads=stoi(argv[5]),K=argc>6?stoi(argv[6]):50;omp_set_num_threads(threads);double start=omp_get_wtime();auto ts=load(argv[1]);{auto b=load(argv[2]);ts.insert(ts.end(),make_move_iterator(b.begin()),make_move_iterator(b.end()));}auto qs=load(argv[3]);cerr<<"loaded "<<ts.size()<<" targets "<<qs.size()<<" queries in "<<omp_get_wtime()-start<<"s\n";
unordered_map<string,int> dict;vector<Term>terms;dict.reserve(5000000);
auto tokenize=[&](Rec&r,bool isTarget,int rid){for(int field=0;field<2;field++){auto &dest=field?r.at:r.nt;for(auto&w:split(field?r.a:r.n)){if(w.empty())continue;string key=r.c+char(field?'A':'N')+w;auto it=dict.find(key);int tid;if(it==dict.end()){tid=terms.size();dict.emplace(key,tid);terms.emplace_back();}else tid=it->second;if(find(dest.begin(),dest.end(),tid)==dest.end()){dest.push_back(tid);if(isTarget)terms[tid].p.push_back(rid);}}}};
for(int i=0;i<(int)ts.size();i++)tokenize(ts[i],true,i);for(auto&q:qs)tokenize(q,false,0);for(auto&t:terms)t.w=log(1.+double(ts.size())/(1+t.p.size()));cerr<<"indexed "<<terms.size()<<" terms in "<<omp_get_wtime()-start<<"s\n";dict.clear();dict.rehash(0);
string prefix=argv[4];{ofstream o(prefix+".targets");for(auto&t:ts)o<<t.id<<'\n';ofstream q(prefix+".queries");for(auto&r:qs)q<<r.id<<'\t'<<r.c<<'\n';}
const int BATCH=2500;int batches=(qs.size()+BATCH-1)/BATCH;
#pragma omp parallel
{
vector<float> ns(ts.size(),0),as(ts.size(),0);vector<int>stamp(ts.size(),-1),touched;
#pragma omp for schedule(dynamic)
for(int batch=0;batch<batches;batch++){string base=prefix+"."+to_string(batch);ofstream out(base+".bin.tmp",ios::binary);int end=min((int)qs.size(),(batch+1)*BATCH);long pairs=0;
for(int qi=batch*BATCH;qi<end;qi++){auto&q=qs[qi];touched.clear();for(int field=0;field<2;field++){auto ids=field?q.at:q.nt;sort(ids.begin(),ids.end(),[&](int a,int b){return terms[a].p.size()<terms[b].p.size();});int used=0;for(int tid:ids){auto&term=terms[tid];if(term.p.empty()||term.p.size()>40000)continue;if(++used>(field?6:4))break;float w=term.w*term.w;for(int ti:term.p){if(stamp[ti]!=qi){stamp[ti]=qi;ns[ti]=as[ti]=0;touched.push_back(ti);}if(field)as[ti]+=w;else ns[ti]+=w;}}}
float maxn=0,maxa=0;for(int ti:touched){maxn=max(maxn,ns[ti]);maxa=max(maxa,as[ti]);}maxn=max(1.f,maxn);maxa=max(1.f,maxa);
vector<int> chosen;auto addtop=[&](int mode,int count){auto score=[&](int t){float n=ns[t]/maxn,a=as[t]/maxa;return mode==0?n:(mode==1?a:1.2f*n+a+1.5f*min(n,a));};auto cmp=[&](int a,int b){float x=score(a),y=score(b);return x==y?a<b:x>y;};int k=min(count,(int)touched.size());partial_sort(touched.begin(),touched.begin()+k,touched.end(),cmp);for(int i=0;i<k;i++)if(find(chosen.begin(),chosen.end(),touched[i])==chosen.end())chosen.push_back(touched[i]);};addtop(2,K);addtop(0,12);addtop(1,12);
int rank=0;for(int ti:chosen){auto f=features(q,ts[ti],terms,ns[ti]/maxn,as[ti]/maxa,rank++);uint32_t u=qi,v=ti;out.write((char*)&u,4);out.write((char*)&v,4);out.write((char*)f.data(),f.size()*4);pairs++;}
}out.close();rename((base+".bin.tmp").c_str(),(base+".bin").c_str());
#pragma omp critical
cerr<<"batch "<<batch<<" / "<<batches<<" pairs "<<pairs<<" seconds "<<omp_get_wtime()-start<<"\n";
}
}
cerr<<"done "<<omp_get_wtime()-start<<"s\n";
}
