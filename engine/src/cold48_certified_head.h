  int cert_ring=getenv("EV_HEAD_RING")?atoi(getenv("EV_HEAD_RING")):0;
  float cert_dist=getenv("EV_HEAD_DIST")?atof(getenv("EV_HEAD_DIST")):28;
  bool cert_block=getenv("EV_HEAD_BLOCK") && atoi(getenv("EV_HEAD_BLOCK"));
  BufRef ctop,cwnmax,cjobs,cjobgrid;
  BufRef cnorm,cold,cxold,cids,cpick,cprops,ccnt,ccand,cmap,corig,cgrid,cx,csmall;
  std::vector<int> ctoken;
  int cert_steps=0,cert_previous[48]{},cert_start=0;
  int64_t cert_full_count=0,cert_single_count=0;
  bool cert_enabled(int t,int seqs,bool greedy) const {
    return cert_ring && greedy && t==48 && seqs==48 && n_out==48 && !verify && !any_ext && !a8_ready &&
      cfg.softcap<=0 && cfg.hidden==5120 && cfg.vocab==248320 && lm_head.bits==4 && !lm_head.ldk && !g.nax;
  }
  void cert_project(Enc& e,const BufRef& hidden,const BufRef& output,const BufRef& predicted,const int* tokens){
    constexpr int k=5120,v=248320,B=48,limit=512;
    const int flip=lm_head.flipped;
    if(!cnorm.buf){
      if(cert_ring<1 || cert_ring>64)abort();ctoken.assign(B*cert_ring,-1);
      cnorm=g.alloc(v*4);cold=g.alloc(size_t(B)*cert_ring*v*2);cxold=g.alloc(size_t(B)*cert_ring*k*2);cids=g.alloc(B*cert_ring*4);
      cpick=g.alloc(B*20);cprops=g.alloc(B*8*4);ccnt=g.alloc(B*4);ccand=g.alloc(B*limit*4);csmall=g.alloc(B*limit*2);
      cmap=g.alloc(B*4);corig=g.alloc(B*4);cgrid=g.alloc(21*4);cx=g.alloc(B*k*2);
      e.k("cert_norm").buf(0,lm_head.wq).buf(1,lm_head.s).buf(2,lm_head.b).buf(3,cnorm).args(4,flip);e.groups(S3(v),S3(128));
      if(cert_block){
        ctop=g.alloc(size_t(B)*cert_ring*(v/256)*2);cwnmax=g.alloc(v/256*4);cjobs=g.alloc(B*(v/256)*4);cjobgrid=g.alloc(12);
        e.k("cert_block_norm").buf(0,cnorm).buf(1,cwnmax);e.groups(S3(v/256),S3(256));
      }
    }
    if(cert_steps){
      for(int i=0;i<B;++i){bool full=cmap.ptr<int>()[i]>=0;ctoken[cert_start+i]=full?cert_previous[i]:-1;
        cert_full_count+=full;cert_single_count+=!full && ccnt.ptr<uint32_t>()[i]==1;}
      if(cert_steps%64==0)fprintf(stderr,"certified head steps %d full %lld single %lld rows %d\n",cert_steps,(long long)cert_full_count,(long long)cert_single_count,B*cert_steps);
    }
    cert_start=(cert_steps%cert_ring)*B;cert_steps++;
    int* proposed=cprops.ptr<int>();std::fill_n(proposed,B*8,-1);
    for(int i=0;i<B;i++){
      int n=0;for(int j=1;j<=int(ctoken.size()) && n<8;j++){
        int q=(cert_start+ctoken.size()-j)%ctoken.size();if(ctoken[q]==tokens[i])proposed[i*8+n++]=q;
      }
      cert_previous[i]=tokens[i];
    }
    e.k("cert_near").buf(0,hidden).buf(1,cxold).buf(2,cprops).buf(3,cids).buf(4,cpick).args(5,cert_dist);e.groups(S3(B),S3(256));
    e.k("cert_incumbent").buf(0,lm_head.wq).buf(1,lm_head.s).buf(2,lm_head.b).buf(3,hidden).buf(4,cpick).buf(5,cnorm).args(6,flip).buf(7,ccnt);e.groups(S3(B),S3(128));
    if(cert_block){
      e.k("cert_reset").buf(0,cjobgrid);e.threads(S3(3),S3(3));
      e.k("cert_blocks").buf(0,ctop).buf(1,cwnmax).buf(2,cpick).buf(3,cjobgrid).buf(4,cjobs);e.threads(S3(992,B),S3(32));
      e.k("cert_filter_blocks").buf(0,cold).buf(1,cnorm).buf(2,cpick).buf(3,ccnt).buf(4,ccand).buf(5,cjobs);
      [e.e dispatchThreadgroupsWithIndirectBuffer:cjobgrid.buf indirectBufferOffset:0 threadsPerThreadgroup:S3(256)];
    }else{
      e.k("cert_filter").buf(0,cold).buf(1,cnorm).buf(2,cpick).buf(3,ccnt).buf(4,ccand);e.threads(S3(v,B),S3(256));
    }
    e.k("cert_plan").buf(0,ccnt).buf(1,cmap).buf(2,corig).buf(3,cgrid);e.threads(S3(1),S3(1));
    e.k("cert_gather").buf(0,hidden).buf(1,cx).buf(2,corig);e.threads(S3(k/2,B),S3(128));
    for(int m=8;m<=48;m+=8){
      std::string name="mlx_qmm_sgw_b4_altrue_"+std::to_string(m)+"x32";
      e.k(g.pso_fc(name,{{40,lm_head.flipped}})).buf(0,lm_head.wq).buf(1,lm_head.s).buf(2,lm_head.b).buf(3,cx).buf(4,output).args(5,k).args(6,v).args(7,m).args(8,k);
      [e.e dispatchThreadgroupsWithIndirectBuffer:cgrid.buf indirectBufferOffset:12*(m/8-1) threadsPerThreadgroup:S3(128)];
    }
    e.k("cert_short").buf(0,lm_head.wq).buf(1,lm_head.s).buf(2,lm_head.b).buf(3,hidden).buf(4,ccnt).buf(5,ccand).buf(6,csmall).args(7,flip);
    [e.e dispatchThreadgroupsWithIndirectBuffer:cgrid.buf indirectBufferOffset:72 threadsPerThreadgroup:S3(32)];
    e.k("cert_finish").buf(0,output).buf(1,ccnt).buf(2,cmap).buf(3,ccand).buf(4,csmall).buf(5,predicted).buf(6,cids).args(7,cert_start);e.groups(S3(B),S3(1024));
    e.k(cert_block?"cert_store_blocks":"cert_store").buf(0,output).buf(1,hidden).buf(2,cmap).buf(3,cold).buf(4,cxold).args(5,cert_start);
    if(cert_block){e.buf(6,ctop);e.groups(S3(v/256,B),S3(128));}
    else e.threads(S3(v/2,B),S3(256));
  }
