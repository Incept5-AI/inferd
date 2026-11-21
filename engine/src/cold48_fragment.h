  struct FragConfig{int layout,slab,width,wm,mixed;};
  struct FragWeights{BufRef codes,meta;};
  struct FragPlan{FragConfig config;FragWeights weights;};
  std::map<const QLinear*,FragPlan> fragment_plans;
  FragWeights ev_pack_fragment(Enc& e,const QLinear& w,FragConfig c){
    if(c.layout==0)return {};
    const size_t padded=(w.N+8*c.slab-1)/(8*c.slab)*(8*c.slab);
    const size_t bytes=padded*w.K/2;
    FragWeights pack{g.alloc(c.layout==2?bytes*4:bytes),g.alloc(c.layout==2?16:padded*(w.K/64)*4)};
    struct{int k,n,flip,slab,dense;} args{w.K,w.N,int(w.flipped),c.slab,int(c.layout==2)};
    e.k("ev_pack_weights").buf(0,w.wq).buf(1,w.s).buf(2,w.b).buf(3,pack.codes).buf(4,pack.meta).args(5,args);
    e.threads(S3(bytes/8),S3(256));return pack;
  }
  void ev_draw_fragment(Enc& e,const BufRef& x,const QLinear& w,const BufRef& y,int mode,FragConfig c,FragWeights pack){
    char name[100];snprintf(name,sizeof(name),"evf_%d_%d_%d_%d_%d_%d_%d",w.K,w.N,c.layout,c.slab,c.width,c.wm,c.mixed);
    struct{int flip,add;} flags{int(w.flipped),mode};
    e.k(name).buf(0,c.layout?pack.codes:w.wq).buf(1,c.layout?pack.meta:w.s).buf(2,w.b).buf(3,x).buf(4,y).args(5,flags);
    int threads=c.wm<10?128:32*(c.wm/10)*(c.wm%10);
    e.groups(S3((w.N+c.width-1)/c.width),S3(threads));
  }
  bool ev_run_fragment(Enc& e,const BufRef& x,const QLinear& w,const BufRef& y,int mode){
    if(fragment_plans.empty())return false;
    auto p=fragment_plans.find(&w);if(p==fragment_plans.end())return false;
    ev_draw_fragment(e,x,w,y,mode,p->second.config,p->second.weights);return true;
  }
  void ev_prepare_fragments(const char* table){
    FILE* f=fopen(table,"r");if(!f)abort();int k,n;FragConfig c;
    std::map<std::pair<int,int>,FragConfig> choose;
    while(fscanf(f,"%d%d%d%d%d%d%d",&k,&n,&c.layout,&c.slab,&c.width,&c.wm,&c.mixed)==7)choose[{k,n}]=c;
    fclose(f);
    auto start=std::chrono::steady_clock::now();size_t bytes=0;std::vector<QLinear*> weights;
    for(auto& l:L)for(auto w:{&l.gate_up,&l.down,&l.in_proj,&l.out_proj})weights.push_back(w);
    weights.push_back(&lm_head);
    for(auto* w:weights){
      auto p=choose.find({w->K,w->N});
      if(p==choose.end() || w->bits!=4 || w->ldk)continue;
      auto cb=[g.queue commandBuffer];Enc enc(g,cb);auto pack=ev_pack_fragment(enc,*w,p->second);
      enc.end();[cb commit];[cb waitUntilCompleted];if(cb.error){fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);abort();}
      bytes+=pack.codes.buf.length+pack.meta.buf.length;
      fragment_plans.emplace(w,FragPlan{p->second,pack});
    }
    fprintf(stderr,"fragment weights %zu additional_GiB %.4f startup_ms %.2f\n",fragment_plans.size(),bytes/double(1ull<<30),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
  }
