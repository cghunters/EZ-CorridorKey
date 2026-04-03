# Демонстрации и ссылки: Keying & Matting Tools

> Актуально на: 2026-04-03

---

## Open-source модели

### CorridorKey
| Ресурс | Ссылка |
|--------|--------|
| GitHub (оригинал) | https://github.com/nikopueringer/CorridorKey |
| GitHub (EZ-CorridorKey GUI) | https://github.com/edenaion/EZ-CorridorKey |
| Nuke plugin | https://github.com/petermercell/CorridorKey-for-Nuke |
| ComfyUI nodes | https://github.com/cnoellert/comfyui-corridorkey |
| Облачный сервис | https://www.corridorkey.app/ |
| Обзор Hackaday | https://hackaday.com/2026/03/18/corridorkey-is-what-you-get-when-artists-make-ai-tools/ |
| Обзор No Film School | https://nofilmschool.com/corridorkey-ai-chroma-keyer |
| DeepWiki | https://deepwiki.com/nikopueringer/CorridorKey |

### GVM (Generative Video Matting)
| Ресурс | Ссылка |
|--------|--------|
| GitHub | https://github.com/aim-uofa/GVM |
| Paper (arXiv) | https://arxiv.org/abs/2508.07905 |
| Project page | https://yongtaoge.github.io/project/gvm |
| SIGGRAPH 2025 | https://dl.acm.org/doi/10.1145/3721238.3730642 |
| Модель (HuggingFace) | https://huggingface.co/geyongtao/gvm |
| Dataset SynHairMan | https://huggingface.co/datasets/geyongtao/SynHairMan |
| Онлайн-демо | ❌ Нет (только локально) |

### SAM2 (Segment Anything Model 2) — Meta
| Ресурс | Ссылка |
|--------|--------|
| GitHub | https://github.com/facebookresearch/sam2 |
| Официальная страница | https://ai.meta.com/sam2/ |
| Paper (arXiv) | https://arxiv.org/abs/2408.00714 |
| **Онлайн-демо** ✅ | https://sam2.metademolab.com/demo |
| HuggingFace | https://huggingface.co/papers/2408.00714 |

### VideoMaMa (Video Mask-to-Matte) — Adobe Research + KAIST
| Ресурс | Ссылка |
|--------|--------|
| GitHub (411 stars) | https://github.com/cvlab-kaist/VideoMaMa |
| Paper (arXiv) | https://arxiv.org/abs/2601.14255 |
| Project page | https://cvlab-kaist.github.io/VideoMaMa/ |
| **HuggingFace демо** ✅ | https://huggingface.co/spaces/SammyLim/VideoMaMa |
| ComfyUI nodes | Есть (с Feb 2026) |
| Лицензия | CC BY-NC 4.0 (код), Stability AI Community (веса) |

### MatAnyone2 — S-Lab (NTU) + SenseTime
| Ресурс | Ссылка |
|--------|--------|
| GitHub | https://github.com/pq-yang/MatAnyone2 |
| GitHub (MatAnyone 1) | https://github.com/pq-yang/MatAnyone |
| Paper (arXiv) | https://arxiv.org/abs/2512.11782 |
| Project page | https://pq-yang.github.io/projects/MatAnyone2/ |
| **HuggingFace демо** ✅ | https://huggingface.co/spaces/PeiqingYang/MatAnyone |
| ComfyUI nodes | https://github.com/FuouM/ComfyUI-MatAnyone |

### BiRefNet (Bilateral Reference Network)
| Ресурс | Ссылка |
|--------|--------|
| GitHub | https://github.com/ZhengPeng7/BiRefNet |
| Paper (arXiv) | https://arxiv.org/abs/2401.03407 |
| Модель (HuggingFace) | https://huggingface.co/ZhengPeng7/BiRefNet |
| Matting-модель | https://huggingface.co/ZhengPeng7/BiRefNet-matting |
| **HuggingFace демо** ✅ | https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo |
| ComfyUI nodes | https://github.com/viperyl/ComfyUI-BiRefNet |

---

## Коммерческие продукты

### Boris FX — Matte Refine ML (NAB 2025)
| Ресурс | Ссылка |
|--------|--------|
| Анонс NAB 2025 | https://blog.borisfx.com/press/nab-2025-explore-groundbreaking-new-ai-plugins-from-boris-fx |
| Видео: hair detail AI masking | https://borisfx.com/videos/silhouette-2025-reveal-fine-hair-detail-with-ai-masking/ |
| Документация | https://borisfx.com/documentation/silhouette-2025.5/silhouette-2025.5/Nodes-Matte%20Refine%20ML.html |
| Mocha Pro 2026 AI roto/matte | https://blog.borisfx.com/press/boris-fx-mocha-pro-adds-powerful-new-ai-roto-matte-and-3d-solve-refinement-tools |

### ONYX AI Matte (OFX plugin для Nuke / Resolve) — ⚠️ Windows + NVIDIA only
| Ресурс | Ссылка |
|--------|--------|
| Сайт | https://onyxofx.com/ |
| Обзор CG Channel | https://www.cgchannel.com/2026/03/onyx-ai-matte-automatically-generates-masks-from-video-footage/ |
| Обзор No Film School | https://nofilmschool.com/onyx-ai-matte-2-5 |
| Обзор 80.lv | https://80.lv/articles/ai-powered-matte-extraction-plug-in-for-davinci-resolve-nuke |
| Logik Forums | https://forum.logik.tv/t/onix-ai-matte/14192 |
| Цена | €80 perpetual (indie), trial 7 дней бесплатно |
| ML-стек | SAM3 (Meta) + VitMatte, ONNX FP16, TensorRT 10.x |
| **Ограничение** | Windows 10+ only, NVIDIA RTX 2060+ (CUDA 12.6), не работает на macOS/Linux |

### Beeble AI — полная экосистема
| Ресурс | Ссылка |
|--------|--------|
| Сайт | https://beeble.ai/ |
| **VFX Passes (app)** | https://app.beeble.ai/vfx-passes |
| Beeble Studio (desktop) | https://beeble.ai/beeble-studio |
| Pricing (Cloud) | https://beeble.ai/pricing-cloud |
| Nuke Plugin docs | https://docs.beeble.ai/integration/nuke-plugin |
| Video-to-VFX docs | https://docs.beeble.ai/beeble/video-to-vfx |
| SwitchX (generative video) | https://docs.beeble.ai/beeble/switchx |
| Beeble Camera (iOS, бесплатно) | https://apps.apple.com/us/app/beeble-camera/id6739796149 |
| CVPR 2024 paper (SwitchLight) | https://arxiv.org/html/2402.18848v1 |
| Superman & Lois showcase | https://beeble.ai/showcase/superman-lois-relighting-vfx |
| Boxel Studio case | https://boxelstudio.com/beeble-switchlight/ |
| Обзор CG Channel | https://www.cgchannel.com/2025/11/beeble-launches-switchlight-3-0/ |
| Обзор RedShark (SwitchX) | https://www.redsharknews.com/beeble-switchx-generative-video-vfx-relighting |
| TechCrunch (funding) | https://techcrunch.com/2024/07/10/beeble-ai-raises-4-75m-to-launch-a-virtual-production-platform-for-indie-filmmakers/ |
| **Ограничение Studio** | Windows/Linux only, NVIDIA RTX 30+ (12GB+), нет macOS |

### Kognat Rotobot (OFX для Nuke)
| Ресурс | Ссылка |
|--------|--------|
| Сайт | https://kognat.com/ |

### Foundry / Nuke AI экосистема
| Ресурс | Ссылка |
|--------|--------|
| Foundry AI Solutions | https://www.foundry.com/ai-solutions |
| Griptape acquisition (официально) | https://www.foundry.com/news-and-awards/foundry-acquires-griptape-ai |
| CG Channel: Griptape | https://www.cgchannel.com/2026/02/foundry-acquires-ai-tools-firm-griptape/ |
| fxguide podcast: Griptape | https://www.fxguide.com/fxpodcasts/foundry-acquires-griptape-an-exclusive-fxpodcast-interview/ |
| Griptape Nodes (визуальный AI builder) | https://www.griptapenodes.com/ |
| Griptape MCP | https://www.griptape.ai/blog/model-context-protocol-mcp-servers-as-tools |
| USD MCP Server (GitHub) | https://github.com/griptape-ai/usd-mcp-server |
| **Nuke 17 release** (BigCat, Gaussian Splats, USD) | https://www.foundry.com/news-and-awards/foundry-releases-nuke-17-advancing-compositing-workflows |
| CG Channel: Nuke 17 | https://www.cgchannel.com/2026/02/foundry-releases-nuke-17-0-nukex-17-0-nuke-studio-17-0/ |
| fxguide: Nuke 17 | https://www.fxguide.com/fxpodcasts/nuke-17-with-foundry-creative-director-juan-salazar/ |
| **Cattery** (каталог .cat нейросетей) | https://community.foundry.com/cattery |
| Cattery guide | https://www.foundry.com/insights/machine-learning/the-artists-guide-to-cattery |
| CopyCat overview | https://www.foundry.com/insights/film-tv/copycat-machine-learning-nuke |
| SmartROTO research | https://www.foundry.com/insights/machine-learning/smartroto-enabling-rotoscoping |
| **Nuke Stage** (VP, NAB 2025 Best of Show) | https://nofilmschool.com/foundry-nuke-stage-nab-2025 |
| fxphd: Nuke MCP курс | https://www.fxphd.com/details/707/ |
| CorridorKey-for-Nuke | https://github.com/petermercell/CorridorKey-for-Nuke |

### MARZ (AI roto cleanup)
| Ресурс | Ссылка |
|--------|--------|
| Сайт | https://www.mfrx.com/ |
| Примечание | 30-40% снижение стоимости рото. Студийный инструмент |

---

*Обновлено: 2026-04-03*
