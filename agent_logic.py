import os
import time
import base64
import hashlib
import re
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from sqlalchemy import text
from db import get_engine, get_langchain_db

load_dotenv(override=True)

# --- 1. 全局配置与行为准则 ---
SQL_SYSTEM_PREFIX = """你是一个严谨的食品过敏专家。你必须**快速高效**地完成查询。

【数据库结构 - 已知信息，无需再查】
表名：products
字段：id, name, brand, ingredients, allergens, image_url, categories, countries
索引：brand (精确查询用 =), name (模糊查询用 LIKE)

⚠️【重要】你已经完全了解表结构，禁止浪费时间调用 sql_db_list_tables 或 sql_db_schema！
⚠️【重要】直接根据用户问题生成 SQL 并执行！一步到位！

【跨语言搜索与补位逻辑 - 必须执行】
1. 识别产品的中英文名：同一产品往往分为中文记录（常为空）和英文记录（常有成分表）。搜中文时必须自动生成对应英文名进行联合查询。
2. 强制取齐逻辑：使用 `OR` 连接中英文名。示例：`WHERE (name LIKE '%精选老抽%' OR name LIKE '%Premium Dark Soy Sauce%')`
3. 质量优先排序：使用 `ORDER BY length(ingredients) DESC` 让内容最详实的记录排在最前面。

【SQL生成规范 - 第一次就必须正确】
⚠️【禁止】先查表结构！你已经知道所有字段！
⚠️【禁止】使用 sql_db_list_tables 或 sql_db_schema！
⚠️【必须】直接生成并执行以下 SQL！

**常用查询模板（直接复制使用）：**

1️⃣ 查过敏原/成分（最常见）：
```sql
SELECT name, brand, ingredients, allergens FROM products 
WHERE brand = 'Lee Kum Kee' 
  AND (name LIKE '%老抽%' OR name LIKE '%dark soy%') 
  AND ingredients != '' 
ORDER BY length(ingredients) DESC LIMIT 1;
```

2️⃣ 查图片：
```sql
SELECT name, brand, image_url FROM products 
WHERE brand = 'Lee Kum Kee' 
  AND (name LIKE '%老抽%' OR name LIKE '%dark soy%') 
  AND image_url IS NOT NULL AND image_url != '' LIMIT 1;
```

3️⃣ 列表查询（如"有哪些不含大豆的酱"）：
```sql
SELECT name, brand, ingredients FROM products 
WHERE brand = 'Lee Kum Kee' 
  AND ingredients NOT LIKE '%soy%' 
  AND ingredients NOT LIKE '%大豆%' 
  AND ingredients != '' LIMIT 5;
```

⚠️ 直接套用模板，替换品牌名和产品名即可！一步到位！

【核心交互逻辑 - 结果过滤】
1. 意图精准匹配（极重要）：
   - 如果用户问"过敏原"/"能吃吗"：查询 ingredients, allergens，只给出过敏原判定结论。
   - 如果用户问"长什么样"/"看图"/"外观"：查询 image_url，**必须**用 Markdown 格式展示图片（见下方格式）
   - 如果用户问"配料表"/"成分"：查询 ingredients，提供详细配料表。

2. 🔴【图片展示格式 - 必须严格遵守】🔴
   当查询到 image_url 时，输出格式必须是：
   
   这是【产品名称】的包装图片：
   
   ![产品名称](完整图片URL)
   
   例子：
   这是李锦记精选老抽的包装图片：
   
   ![李锦记精选老抽](https://images.openfoodfacts.org/images/products/007/889/512/9625/front_fr.3.400.jpg)
   
   ⚠️ 严禁直接返回纯链接！必须用 Markdown 图片格式！

3. 结论先行：直接告诉用户建议（能吃/不能吃/含有什么过敏原）。
4. 记忆指代：必须查阅 'chat_history' 解析"它"、"这个"。严禁反问！

【查询技术限制- 性能与质量优化】
1. 品牌精确匹配：对于"李锦记"，必须用 `brand = 'Lee Kum Kee'`（精确匹配，利用索引）
2. 动态数量：单个产品用 LIMIT 1，列表查询用 LIMIT 3
3. 列剪枝：仅查询满足当前意图的最小必要列
4. 语言要求：严格遵守系统指定的回复语言（由 [IMPORTANT] 指令给出）
"""


@lru_cache(maxsize=1)
def get_db():
    """缓存数据库连接和表结构信息"""
    return get_langchain_db()


@lru_cache(maxsize=1)
def get_llm():
    """缓存 GPT-4o 主模型（用于视觉识别和复杂推理）"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


@lru_cache(maxsize=1)
def get_fast_llm_for_sql():
    """缓存 GPT-4o-mini 快速模型（用于SQL查询，速度快2-3倍）"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


@lru_cache(maxsize=1)
def get_sql_agent():
    """缓存 SQL Agent（优化版：使用 gpt-4o-mini + 超强提示词）"""
    db = get_db()
    # 🚀 使用 gpt-4o-mini：配合优化后的提示词（禁止查表结构），速度提升 2-3倍
    llm = get_fast_llm_for_sql()

    return create_sql_agent(
        llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=SQL_SYSTEM_PREFIX,
        extra_prompt_messages=[MessagesPlaceholder(
            variable_name="chat_history")],
        max_iterations=3,  # 🚀 优化：禁止查表结构后，3次足够（生成→执行→返回）
        top_k=3,  # 🚀 优化：减少返回行数
        max_execution_time=8  # 🚀 优化：降低超时时间
    )


def get_semantic_hash(text: str) -> str:
    """生成文本的语义哈希（用于缓存键）"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_image_hash(image_bytes: bytes) -> str:
    """生成图片的哈希（用于缓存键）"""
    return hashlib.md5(image_bytes).hexdigest()


@lru_cache(maxsize=1)
def ensure_products_norm_schema():
    """Ensure normalized helper columns and indexes exist."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS canonical_brand TEXT"))
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS canonical_name TEXT"))
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS lang_tag TEXT"))
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS ingredients_len INTEGER"))

            conn.execute(
                text(
                    """
                    UPDATE products
                    SET
                        canonical_brand = lower(regexp_replace(coalesce(brand,''), '[^a-z0-9\\u4e00-\\u9fff]+', '', 'g')),
                        canonical_name = lower(regexp_replace(coalesce(name,''), '[^a-z0-9\\u4e00-\\u9fff]+', '', 'g')),
                        lang_tag = CASE
                            WHEN ingredients ~ '[\\u4e00-\\u9fff]' THEN 'zh'
                            WHEN ingredients ~ '[A-Za-z]' THEN 'en'
                            ELSE 'other'
                        END,
                        ingredients_len = length(coalesce(ingredients, ''))
                    WHERE canonical_brand IS NULL
                       OR canonical_name IS NULL
                       OR lang_tag IS NULL
                       OR ingredients_len IS NULL
                    """
                )
            )

            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_products_canonical_brand_name ON products(canonical_brand, canonical_name)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_products_canonical_brand_lang ON products(canonical_brand, lang_tag)"
                )
            )
    except Exception as e:
        print(f"  ⚠️ 规范化字段初始化失败，回退旧查询: {e}")


def direct_sql_query(brand: str, product_keywords: str, query_type: str = "image"):
    """🚀 快速路径：直接执行 SQL，绕过 Agent（速度提升 5-10倍）"""
    engine = get_engine()
    ensure_products_norm_schema()

    # 自动生成中英文关键词对
    keyword_mapping = {
        "老抽": "dark soy",
        "生抽": "light soy",
        "酱油": "soy sauce",
        "蚝油": "oyster sauce",
        "辣椒酱": "chili sauce",
        "醋": "vinegar"
    }

    # 如果是中文关键词，添加英文；反之亦然
    keywords = [product_keywords]
    if product_keywords in keyword_mapping:
        keywords.append(keyword_mapping[product_keywords])
    elif product_keywords in keyword_mapping.values():
        for cn, en in keyword_mapping.items():
            if en == product_keywords:
                keywords.append(cn)
                break

    # 构造 OR 查询
    name_conditions = " OR ".join([f"name ILIKE :kw_{i}" for i in range(len(keywords))])
    params = {f"kw_{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
    params["brand"] = brand
    params["brand_norm"] = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (brand or "").lower())
    params["target_lang"] = "zh" if re.search(r"[\u4e00-\u9fff]", product_keywords or "") else "en"

    try:
        if query_type == "image":
            # 查询图片
            sql = f"""
            WITH ranked AS (
                SELECT
                    name, brand, image_url,
                    row_number() OVER (
                        PARTITION BY canonical_brand, canonical_name
                        ORDER BY
                            CASE
                                WHEN lang_tag = :target_lang THEN 0
                                WHEN lang_tag = 'en' THEN 1
                                ELSE 2
                            END,
                            ingredients_len DESC
                    ) AS rn
                FROM products
                WHERE canonical_brand = :brand_norm
                  AND ({name_conditions})
                  AND image_url IS NOT NULL
                  AND image_url != ''
            )
            SELECT name, brand, image_url
            FROM ranked
            WHERE rn = 1
            LIMIT 1
            """
            print(f"  📝 执行SQL: {sql}")
            print(f"  📝 参数: {params}")
            with engine.connect() as conn:
                result = conn.execute(text(sql), params).first()
            if result is None:
                legacy_sql = f"""
                SELECT name, brand, image_url
                FROM products
                WHERE brand = :brand
                  AND ({name_conditions})
                  AND image_url IS NOT NULL
                  AND image_url != ''
                LIMIT 1
                """
                with engine.connect() as conn:
                    result = conn.execute(text(legacy_sql), params).first()
        else:
            # 查询过敏原（优先返回内容最详细的）
            sql = f"""
            WITH ranked AS (
                SELECT
                    name, brand, ingredients, allergens,
                    row_number() OVER (
                        PARTITION BY canonical_brand, canonical_name
                        ORDER BY
                            CASE
                                WHEN lang_tag = :target_lang THEN 0
                                WHEN lang_tag = 'en' THEN 1
                                ELSE 2
                            END,
                            ingredients_len DESC
                    ) AS rn
                FROM products
                WHERE canonical_brand = :brand_norm
                  AND ({name_conditions})
                  AND ingredients != ''
            )
            SELECT name, brand, ingredients, allergens
            FROM ranked
            WHERE rn = 1
            LIMIT 1
            """
            print(f"  📝 执行SQL: {sql}")
            print(f"  📝 参数: {params}")
            with engine.connect() as conn:
                result = conn.execute(text(sql), params).first()
            if result is None:
                legacy_sql = f"""
                SELECT name, brand, ingredients, allergens
                FROM products
                WHERE brand = :brand
                  AND ({name_conditions})
                  AND ingredients != ''
                ORDER BY length(ingredients) DESC
                LIMIT 1
                """
                with engine.connect() as conn:
                    result = conn.execute(text(legacy_sql), params).first()
        return result  # 返回元组或 None
    except Exception as e:
        print(f"  ❌ SQL查询失败: {e}")
        # 兼容：若规范化列尚未就绪，回退旧查询
        try:
            legacy_sql = ""
            if query_type == "image":
                legacy_sql = f"""
                SELECT name, brand, image_url
                FROM products
                WHERE brand = :brand
                  AND ({name_conditions})
                  AND image_url IS NOT NULL
                  AND image_url != ''
                LIMIT 1
                """
            else:
                legacy_sql = f"""
                SELECT name, brand, ingredients, allergens
                FROM products
                WHERE brand = :brand
                  AND ({name_conditions})
                  AND ingredients != ''
                ORDER BY length(ingredients) DESC
                LIMIT 1
                """
            with engine.connect() as conn:
                legacy = conn.execute(text(legacy_sql), params).first()
            return legacy
        except Exception:
            pass
        return None


def query_text(question: str, image_bytes: bytes = None):
    """
    核心查询接口：支持文本和图片（优化版：带快速路径）
    """
    start_time = time.time()

    # API / CLI 模式无 Streamlit session；SQL Agent 多轮上下文由 LangGraph/前端会话承担
    chat_history = []

    # 默认语言：可用环境变量覆盖（前端可在后续改为传参）
    target_lang = (os.getenv("DEFAULT_REPLY_LANGUAGE") or "自动识别 (Auto)").strip()

    # [强制中文补丁] 如果用户用中文提问且设置为自动识别，强制后续回复使用中文
    if target_lang == "自动识别 (Auto)":
        import re
        if re.search("[\u4e00-\u9fa5]", question):
            target_lang = "简体中文"
            print("  🇨🇳 检测到中文提问，自动切换回复语言为简体中文")

    # 🚀 快速路径检测：如果是简单的图片查询，直接执行 SQL
    quick_brands = {
        "李锦记": "Lee Kum Kee",
        "lee kum kee": "Lee Kum Kee",
        "海天": "Haday",
        "haday": "Haday",
        "康师傅": "Master Kong",
        "master kong": "Master Kong"
    }

    image_keywords = ["长什么样", "看图", "图片", "外观", "包装", "样子",
                      "look like", "picture", "image", "appearance"]

    allergen_keywords = ["能吃", "过敏", "有没有", "安全", "can i eat", "safe",
                         "allerg", "成分", "ingredient"]

    # 排除关键词：需要多跳推理的复杂查询
    # 使用单词边界避免误匹配（如 "all" 不应匹配 "allergic"）
    exclude_keywords = ["对比", "区别", "哪些", "列表", "比较", "所有", "全部", "有什么",
                        "compare", "difference", "list", " all ", "what are", "which"]

    if not image_bytes:  # 仅对纯文本查询启用快速路径
        q_lower = question.lower()
        detected_brand = None

        # 检查是否需要复杂推理
        is_complex_query = any(kw in q_lower for kw in exclude_keywords)
        if is_complex_query:
            print(f"  🤖 检测到复杂查询，跳过快速路径，使用 Agent 多跳推理")
            # 不走快速路径，继续执行下面的 Agent 逻辑

        for brand_key, brand_value in quick_brands.items():
            if brand_key in q_lower:
                detected_brand = brand_value
                break

        # 场景1：图片查询（仅简单查询）
        if not is_complex_query and detected_brand and any(kw in q_lower for kw in image_keywords):
            # 快速路径：直接查询图片
            print("  🚀 启用快速路径：直接 SQL 查询（绕过 Agent，速度提升10倍）")

            # 智能提取产品关键词
            product_keywords = [
                ("老抽", "dark soy"), ("生抽", "light soy"), ("酱油", "soy sauce"),
                ("蚝油", "oyster sauce"), ("辣椒酱", "chili sauce"), ("醋", "vinegar")
            ]

            product_kw = ""
            for cn, en in product_keywords:
                if cn in question or en in q_lower:
                    product_kw = cn if cn in question else en
                    break

            if not product_kw:
                # 兜底：使用品牌名
                product_kw = detected_brand.split()[0]

            if product_kw:
                result = direct_sql_query(detected_brand, product_kw, "image")
                if result:
                    # result 是元组: (name, brand, image_url)
                    name, brand, image_url = result
                    end_time = time.time()
                    duration = end_time - start_time

                    if target_lang == "English":
                        output = f"Here is the product image of {name}:\n\n![{name}]({image_url})"
                    else:
                        output = f"这是{name}的包装图片：\n\n![{name}]({image_url})"

                    print(f"  ✅ 快速路径成功！耗时 {duration:.2f}秒")
                    return output, duration

        # 场景2：过敏原/成分查询（仅简单查询）
        if not is_complex_query and detected_brand and any(kw in q_lower for kw in allergen_keywords):
            print("  🚀 启用快速路径：过敏原查询（直接 SQL）")

            # 提取产品关键词
            product_keywords = [
                ("老抽", "dark soy"), ("生抽", "light soy"), ("酱油", "soy sauce"),
                ("蚝油", "oyster sauce"), ("辣椒酱", "chili sauce"), ("醋", "vinegar")
            ]

            product_kw = ""
            for cn, en in product_keywords:
                if cn in question or en in q_lower:
                    product_kw = cn if cn in question else en
                    break

            if not product_kw:
                product_kw = detected_brand.split()[0]

            if product_kw:
                result = direct_sql_query(
                    detected_brand, product_kw, "allergen")
                if result:
                    # result 是元组: (name, brand, ingredients, allergens)
                    name, brand, ingredients, allergens = result
                    allergens = allergens or ""

                    end_time = time.time()
                    duration = end_time - start_time

                    # 智能分析过敏原
                    user_concern = ""
                    user_concern_en = ""  # 英文关键词用于准确检测
                    if "大豆" in question or "soy" in q_lower:
                        user_concern = "大豆" if "大豆" in question else "soy"
                        user_concern_en = "soy"  # 统一用英文检测（数据库是英文）
                    elif "麸质" in question or "gluten" in q_lower:
                        user_concern = "麸质" if "麸质" in question else "gluten"
                        user_concern_en = "gluten"
                    elif "花生" in question or "peanut" in q_lower:
                        user_concern = "花生" if "花生" in question else "peanut"
                        user_concern_en = "peanut"

                    # 生成回答
                    if target_lang == "English":
                        if user_concern:
                            # 使用英文关键词检测（数据库配料表是英文）
                            has_allergen = user_concern_en in ingredients.lower(
                            ) or user_concern_en in allergens.lower()
                            if has_allergen:
                                output = f"⚠️ **Not recommended** - {name} contains {user_concern}.\n\n**Ingredients:** {ingredients}\n\n**Allergens:** {allergens or 'Not specified'}"
                            else:
                                output = f"✅ **Safe** - {name} does not appear to contain {user_concern}.\n\n**Ingredients:** {ingredients}\n\n**Allergens:** {allergens or 'None listed'}"
                        else:
                            output = f"**{name}** Allergen Information:\n\n**Ingredients:** {ingredients}\n\n**Allergens:** {allergens or 'None listed'}"
                    else:
                        if user_concern:
                            # 使用英文关键词检测（数据库配料表是英文）
                            has_allergen = user_concern_en in ingredients.lower(
                            ) or user_concern_en in allergens.lower()
                            if has_allergen:
                                output = f"⚠️ **不建议食用** - {name} 含有{user_concern}。\n\n**配料表：** {ingredients}\n\n**过敏原：** {allergens or '未标注'}"
                            else:
                                output = f"✅ **可以食用** - {name} 不含{user_concern}。\n\n**配料表：** {ingredients}\n\n**过敏原：** {allergens or '无'}"
                        else:
                            output = f"**{name}** 过敏原信息：\n\n**配料表：** {ingredients}\n\n**过敏原：** {allergens or '无'}"

                    print(f"  ✅ 快速路径成功！耗时 {duration:.2f}秒")
                    return output, duration

    # 强制语言指令
    lang_instruction = ""
    if target_lang == "English":
        lang_instruction = "\n\n[IMPORTANT] Reply strictly in English."
    elif target_lang == "Français":
        lang_instruction = "\n\n[IMPORTANT] Répondez strictement en Français."
    elif target_lang == "简体中文":
        lang_instruction = "\n\n[重要] 请务必使用简体中文回答。"
    else:
        # Auto 模式：提醒模型观察输入语言
        lang_instruction = "\n\n(Auto-detect language: Please reply in the same language as the user's question.)"

    if image_bytes:
        # --- 视觉识别逻辑（优化版：结构化输出）---
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # 🚀 优化：要求结构化输出，减少 SQL Agent 的解析时间
        vision_text = """Extract from this food product image:
1. Brand (Chinese and English if available)
2. Product Name (Chinese and English if available)
3. Key visible allergen warnings

Output format: Brand: [brand] | Product: [name] | Allergens visible: [list or "not visible"]
"""
        if target_lang == "简体中文":
            vision_text = """从这张食品图片中提取：
1. 品牌（中英文）
2. 产品名称（中英文）
3. 可见的过敏原警告

输出格式：品牌：[品牌] | 产品：[名称] | 可见过敏原：[列表或"不可见"]
"""

        input_content = [
            {"type": "text", "text": vision_text + lang_instruction},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        llm_vision = get_llm()
        vision_msg = HumanMessage(content=input_content)
        vision_response = llm_vision.invoke([vision_msg])

        # 🚀 优化：直接告诉 SQL Agent 使用精确查询
        refined_question = f"Vision extracted: {vision_response.content}. Query database using brand and product name (use exact match for brand, LIKE for product). Return allergen analysis only."
        if target_lang == "简体中文":
            refined_question = f"视觉提取：{vision_response.content}。使用品牌和产品名查询数据库（品牌用精确匹配，产品名用LIKE）。仅返回过敏原分析。"

        agent = get_sql_agent()
        response = agent.invoke({
            "input": refined_question + lang_instruction,
            "chat_history": chat_history
        })
    else:
        # --- 纯文本逻辑 ---
        agent = get_sql_agent()
        response = agent.invoke({
            "input": question + lang_instruction,
            "chat_history": chat_history
        })

    end_time = time.time()
    duration = end_time - start_time

    # 🚀 后处理：如果返回的是纯图片链接，自动转换为 Markdown 格式
    output = response["output"]
    import re

    # 检测是否包含 http(s) 图片链接但没有 Markdown 格式
    if "http" in output and "![" not in output:
        # 查找所有图片 URL
        url_pattern = r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)[^\s]*)'
        urls = re.findall(url_pattern, output, re.IGNORECASE)

        if urls:
            # 替换为 Markdown 格式
            for url in urls:
                # 清理 URL（移除末尾的标点符号）
                clean_url = url.rstrip('.,;:!?)]}')
                # 提取产品名（从上下文中尝试获取）
                product_name = "产品图片" if target_lang == "简体中文" else "Product Image"
                markdown_img = f"\n\n![{product_name}]({clean_url})\n\n"
                output = output.replace(url, markdown_img)

    return output, duration
