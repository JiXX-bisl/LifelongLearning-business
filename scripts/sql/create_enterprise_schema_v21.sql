-- 虚拟企业数据集 V2.1：MySQL 8.0 六表结构
-- 本脚本不删除既有数据库或表。已有同名表时会保留原表。

CREATE DATABASE IF NOT EXISTS synthetic_enterprise_v21
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE synthetic_enterprise_v21;

CREATE TABLE IF NOT EXISTS enterprise_basic (
  enterprise_id VARCHAR(20) NOT NULL,
  enterprise_name VARCHAR(120) NOT NULL,
  synthetic_credit_code VARCHAR(24) NOT NULL,
  enterprise_type ENUM(
    '有限责任公司', '股份有限公司', '个人独资企业',
    '合伙企业', '外商投资企业', '其他'
  ) NOT NULL,
  registration_status ENUM('存续', '停业', '注销', '吊销', '迁出') NOT NULL,
  industry ENUM(
    '制造', '批发零售', '信息技术', '商务服务', '建筑', '物流运输',
    '金融服务', '房地产', '餐饮住宿', '科研技术', '其他'
  ) NOT NULL,
  registered_capital DECIMAL(18,2) NOT NULL,
  established_date DATE NOT NULL,
  registered_address VARCHAR(255) NOT NULL,
  registered_email VARCHAR(120) NULL,
  registered_phone VARCHAR(30) NULL,
  registered_business_scope TEXT NOT NULL,
  main_business_description TEXT NOT NULL,
  PRIMARY KEY (enterprise_id),
  UNIQUE KEY uq_enterprise_name (enterprise_name),
  UNIQUE KEY uq_synthetic_credit_code (synthetic_credit_code),
  CONSTRAINT chk_basic_id CHECK (enterprise_id REGEXP '^SYN-E[0-9]{6}$'),
  CONSTRAINT chk_basic_credit_code CHECK (synthetic_credit_code REGEXP '^SYN-USCC-[0-9]{6}$'),
  CONSTRAINT chk_basic_capital CHECK (registered_capital BETWEEN 100000.00 AND 2000000000.00),
  CONSTRAINT chk_basic_established_date CHECK (established_date BETWEEN '1990-01-01' AND '2026-08-31'),
  CONSTRAINT chk_basic_name_length CHECK (CHAR_LENGTH(enterprise_name) BETWEEN 4 AND 30),
  CONSTRAINT chk_basic_address_length CHECK (CHAR_LENGTH(registered_address) BETWEEN 1 AND 255),
  CONSTRAINT chk_basic_email CHECK (
    registered_email IS NULL OR
    (CHAR_LENGTH(registered_email) <= 120 AND registered_email LIKE '%@%.example.test')
  ),
  CONSTRAINT chk_basic_phone CHECK (
    registered_phone IS NULL OR registered_phone REGEXP '^(SYN-TEL-[0-9]{6}|021-0000[0-9]{4})$'
  ),
  CONSTRAINT chk_basic_scope_length CHECK (CHAR_LENGTH(registered_business_scope) BETWEEN 50 AND 500),
  CONSTRAINT chk_basic_main_desc_length CHECK (CHAR_LENGTH(main_business_description) BETWEEN 50 AND 300)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS enterprise_person (
  person_id VARCHAR(20) NOT NULL,
  enterprise_id VARCHAR(20) NOT NULL,
  person_name VARCHAR(50) NOT NULL,
  age TINYINT UNSIGNED NOT NULL,
  household_region VARCHAR(50) NOT NULL,
  role ENUM(
    '法定代表人', '董事', '监事', '总经理', '副总经理',
    '财务负责人', '核心技术人员', '其他高管'
  ) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NULL,
  is_current TINYINT(1) NOT NULL,
  PRIMARY KEY (person_id),
  UNIQUE KEY uq_person_enterprise (person_id, enterprise_id),
  KEY idx_person_enterprise_role (enterprise_id, role, is_current),
  CONSTRAINT fk_person_enterprise
    FOREIGN KEY (enterprise_id) REFERENCES enterprise_basic (enterprise_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT chk_person_id CHECK (person_id REGEXP '^SYN-P[0-9]{6}$'),
  CONSTRAINT chk_person_name_length CHECK (CHAR_LENGTH(person_name) BETWEEN 2 AND 4),
  CONSTRAINT chk_person_age CHECK (age BETWEEN 22 AND 70),
  CONSTRAINT chk_person_start_date CHECK (start_date <= '2026-08-31'),
  CONSTRAINT chk_person_end_date CHECK (
    end_date IS NULL OR (end_date BETWEEN start_date AND '2026-08-31')
  ),
  CONSTRAINT chk_person_current CHECK (
    (is_current = 1 AND end_date IS NULL) OR
    (is_current = 0 AND end_date IS NOT NULL)
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS enterprise_operation_profile (
  enterprise_id VARCHAR(20) NOT NULL,
  as_of_date DATE NOT NULL,
  has_wechat_account TINYINT(1) NOT NULL,
  wechat_content_summary TEXT NULL,
  total_assets DECIMAL(18,2) NOT NULL,
  total_liabilities DECIMAL(18,2) NOT NULL,
  PRIMARY KEY (enterprise_id),
  CONSTRAINT fk_profile_enterprise
    FOREIGN KEY (enterprise_id) REFERENCES enterprise_basic (enterprise_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT chk_profile_date CHECK (as_of_date = '2026-08-31'),
  CONSTRAINT chk_profile_wechat CHECK (
    (has_wechat_account = 0 AND wechat_content_summary IS NULL) OR
    (has_wechat_account = 1 AND wechat_content_summary IS NOT NULL)
  ),
  CONSTRAINT chk_profile_wechat_length CHECK (
    wechat_content_summary IS NULL OR CHAR_LENGTH(wechat_content_summary) <= 1000
  ),
  CONSTRAINT chk_profile_assets CHECK (total_assets BETWEEN 0.00 AND 20000000000.00),
  CONSTRAINT chk_profile_liabilities CHECK (total_liabilities BETWEEN 0.00 AND 20000000000.00)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS enterprise_operation_monthly (
  enterprise_id VARCHAR(20) NOT NULL,
  stat_month DATE NOT NULL,
  employee_count INT UNSIGNED NOT NULL,
  social_security_count INT UNSIGNED NOT NULL,
  social_security_amount DECIMAL(18,2) NOT NULL,
  tax_amount DECIMAL(18,2) NOT NULL,
  invoice_purchase_amount DECIMAL(18,2) NOT NULL,
  invoice_sales_amount DECIMAL(18,2) NOT NULL,
  utility_invoice_amount DECIMAL(18,2) NOT NULL,
  travel_hospitality_invoice_amount DECIMAL(18,2) NOT NULL,
  operating_revenue DECIMAL(18,2) NOT NULL,
  main_business_revenue DECIMAL(18,2) NOT NULL,
  total_profit DECIMAL(18,2) NOT NULL,
  net_profit DECIMAL(18,2) NOT NULL,
  PRIMARY KEY (enterprise_id, stat_month),
  KEY idx_monthly_stat_month (stat_month),
  CONSTRAINT fk_monthly_enterprise
    FOREIGN KEY (enterprise_id) REFERENCES enterprise_basic (enterprise_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT chk_monthly_month_start CHECK (
    stat_month BETWEEN '2025-09-01' AND '2026-08-01' AND DAY(stat_month) = 1
  ),
  CONSTRAINT chk_monthly_employee CHECK (employee_count BETWEEN 0 AND 20000),
  CONSTRAINT chk_monthly_social_count CHECK (social_security_count BETWEEN 0 AND 20000),
  CONSTRAINT chk_monthly_social_amount CHECK (social_security_amount BETWEEN 0.00 AND 50000000.00),
  CONSTRAINT chk_monthly_tax CHECK (tax_amount BETWEEN 0.00 AND 100000000.00),
  CONSTRAINT chk_monthly_purchase_invoice CHECK (invoice_purchase_amount BETWEEN 0.00 AND 3000000000.00),
  CONSTRAINT chk_monthly_sales_invoice CHECK (invoice_sales_amount BETWEEN 0.00 AND 3000000000.00),
  CONSTRAINT chk_monthly_utility_invoice CHECK (utility_invoice_amount BETWEEN 0.00 AND 50000000.00),
  CONSTRAINT chk_monthly_travel_invoice CHECK (travel_hospitality_invoice_amount BETWEEN 0.00 AND 50000000.00),
  CONSTRAINT chk_monthly_operating_revenue CHECK (operating_revenue BETWEEN 0.00 AND 3000000000.00),
  CONSTRAINT chk_monthly_main_revenue CHECK (
    main_business_revenue BETWEEN 0.00 AND 3000000000.00 AND
    main_business_revenue <= operating_revenue
  ),
  CONSTRAINT chk_monthly_invoice_subitems CHECK (
    utility_invoice_amount + travel_hospitality_invoice_amount <= invoice_purchase_amount
  ),
  CONSTRAINT chk_monthly_total_profit CHECK (total_profit BETWEEN -100000000.00 AND 500000000.00),
  CONSTRAINT chk_monthly_net_profit CHECK (net_profit BETWEEN -100000000.00 AND 500000000.00)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS enterprise_operation_event (
  operation_event_id VARCHAR(24) NOT NULL,
  enterprise_id VARCHAR(20) NOT NULL,
  event_date DATE NOT NULL,
  event_type ENUM('融资', '债券发行', '银行借款', '借款逾期', '专利申请', '专利授权') NOT NULL,
  amount DECIMAL(18,2) NULL,
  valuation DECIMAL(18,2) NULL,
  financing_stage ENUM('种子轮', '天使轮', 'A轮', 'B轮', 'C轮及以上', '战略融资') NULL,
  term_months SMALLINT UNSIGNED NULL,
  quantity INT UNSIGNED NULL,
  event_status ENUM(
    '进行中', '已完成', '已终止', '正常还款',
    '已逾期', '已结清', '已授权', '未授权'
  ) NOT NULL,
  description VARCHAR(500) NOT NULL,
  PRIMARY KEY (operation_event_id),
  KEY idx_operation_event_query (enterprise_id, event_date, event_type),
  CONSTRAINT fk_operation_event_enterprise
    FOREIGN KEY (enterprise_id) REFERENCES enterprise_basic (enterprise_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT chk_operation_event_id CHECK (operation_event_id REGEXP '^SYN-OE[0-9]{8}$'),
  CONSTRAINT chk_operation_event_date CHECK (event_date BETWEEN '2025-09-01' AND '2026-08-31'),
  CONSTRAINT chk_operation_event_amount CHECK (
    amount IS NULL OR amount BETWEEN 10000.00 AND 100000000000.00
  ),
  CONSTRAINT chk_operation_event_valuation CHECK (
    valuation IS NULL OR valuation BETWEEN 1000000.00 AND 100000000000.00
  ),
  CONSTRAINT chk_operation_event_term CHECK (term_months IS NULL OR term_months BETWEEN 1 AND 120),
  CONSTRAINT chk_operation_event_quantity CHECK (quantity IS NULL OR quantity BETWEEN 1 AND 1000),
  CONSTRAINT chk_operation_event_description CHECK (CHAR_LENGTH(description) BETWEEN 20 AND 500),
  CONSTRAINT chk_operation_event_fields CHECK (
    (event_type = '融资' AND amount IS NOT NULL AND valuation IS NOT NULL AND financing_stage IS NOT NULL
      AND term_months IS NULL AND quantity IS NULL) OR
    (event_type = '债券发行' AND amount IS NOT NULL AND term_months IS NOT NULL
      AND valuation IS NULL AND financing_stage IS NULL AND quantity IS NULL) OR
    (event_type IN ('银行借款', '借款逾期') AND amount IS NOT NULL AND term_months IS NOT NULL
      AND valuation IS NULL AND financing_stage IS NULL AND quantity IS NULL) OR
    (event_type IN ('专利申请', '专利授权') AND quantity IS NOT NULL
      AND amount IS NULL AND valuation IS NULL AND financing_stage IS NULL AND term_months IS NULL)
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS enterprise_negative_event (
  negative_event_id VARCHAR(24) NOT NULL,
  enterprise_id VARCHAR(20) NOT NULL,
  related_person_id VARCHAR(20) NULL,
  event_date DATE NOT NULL,
  event_type ENUM('司法纠纷', '投诉', '报警', '企业涉案', '高管涉案') NOT NULL,
  case_category VARCHAR(50) NULL,
  description VARCHAR(1000) NOT NULL,
  handling_result ENUM('待处理', '调查中', '已撤回', '调解', '已处罚', '已判决', '查无实据', '其他') NOT NULL,
  PRIMARY KEY (negative_event_id),
  KEY idx_negative_event_query (enterprise_id, event_date, event_type),
  KEY idx_negative_related_person (related_person_id, enterprise_id),
  CONSTRAINT fk_negative_event_enterprise
    FOREIGN KEY (enterprise_id) REFERENCES enterprise_basic (enterprise_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT fk_negative_event_person
    FOREIGN KEY (related_person_id, enterprise_id)
    REFERENCES enterprise_person (person_id, enterprise_id),
  CONSTRAINT chk_negative_event_id CHECK (negative_event_id REGEXP '^SYN-NE[0-9]{8}$'),
  CONSTRAINT chk_negative_event_date CHECK (event_date BETWEEN '2016-01-01' AND '2026-08-31'),
  CONSTRAINT chk_negative_event_description CHECK (CHAR_LENGTH(description) BETWEEN 20 AND 1000),
  CONSTRAINT chk_negative_event_related_person CHECK (
    event_type <> '高管涉案' OR related_person_id IS NOT NULL
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
