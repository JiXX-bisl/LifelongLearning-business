"""Private operating and funding drivers used before reporting observations.

Amounts follow quantities, contract prices, staff and independently chosen
spending schedules. No target profit or target debt ratio is an input.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

from .common import decimal, money, month_end, rng

FAMILIES = (2, 8, 14, 20, 26, 32)
EQUAL_FAMILIES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38)
def cost_plan(profile, industry):
    """Separate purchased inputs from work performed by the enterprise staff."""
    model = profile["business_model"]
    plans = {
        "小批量定制制造": ((".50", ".62", ".025", ".045"), (".035", ".060"), "metal_materials_and_selected_external_processing"),
        "订单制造": ((".55", ".66", ".025", ".040"), (".030", ".050"), "production_parts_materials_and_specialist_processing"),
        "批量生产": ((".62", ".71", ".030", ".050"), (".030", ".050"), "batch_raw_materials_packaging_and_external_finishing"),
        "规模化制造+渠道销售": ((".65", ".73", ".030", ".050"), (".035", ".055"), "manufacturing_inputs_and_external_channel_services"),
        "多产品订单制造": ((".57", ".68", ".030", ".050"), (".030", ".055"), "retained_product_materials_and_outside_processing"),
        "电商零售": ((".55", ".65", ".025", ".040"), (".060", ".100"), "retail_sku_purchase_and_platform_fees"),
        "批发分销": ((".80", ".86", ".020", ".035"), (".025", ".045"), "wholesale_goods_and_external_distribution_services"),
        "区域经销": ((".78", ".84", ".020", ".035"), (".025", ".045"), "regional_goods_purchase_and_outside_logistics"),
        "门店+电商经营": ((".60", ".72", ".035", ".060"), (".045", ".075"), "retail_sku_purchase_and_channel_fees"),
        "多区域批发零售网络": ((".76", ".85", ".025", ".045"), (".035", ".055"), "regional_goods_mix_and_outside_channel_services"),
        "软件项目开发": ((".22", ".34", ".035", ".055"), (".035", ".055"), "own_development_staff_with_external_components_and_selected_subcontracts"),
        "SaaS订阅服务": ((".10", ".19", ".040", ".065"), (".045", ".075"), "cloud_component_licenses_and_external_security_services"),
        "软件开发+运维服务": ((".30", ".46", ".035", ".060"), (".040", ".065"), "third_party_components_and_specialist_project_subcontracts"),
        "企业软件平台服务": ((".13", ".23", ".055", ".085"), (".045", ".070"), "platform_cloud_resources_components_and_external_specialist_services"),
        "系统集成与定制开发": ((".50", ".64", ".040", ".065"), (".045", ".070"), "system_hardware_components_and_specialist_integrator_subcontracts"),
        "咨询项目制": ((".08", ".16", ".030", ".050"), (".040", ".065"), "external_data_research_and_selected_expert_services"),
        "常年顾问服务": ((".06", ".14", ".030", ".050"), (".035", ".060"), "advisory_data_licenses_and_external_specialist_opinions"),
        "企业外包服务": ((".08", ".18", ".035", ".055"), (".035", ".060"), "customer_service_platform_and_external_support_inputs_not_own_seat_labor"),
        "营销会展项目制": ((".52", ".68", ".025", ".045"), (".050", ".080"), "external_venues_event_production_and_media_purchases"),
        "专业咨询项目制": ((".08", ".18", ".030", ".050"), (".040", ".070"), "external_professional_data_and_specialist_review_services"),
        "专业分包": ((".45", ".58", ".025", ".045"), (".035", ".060"), "specialty_construction_materials_and_external_tool_hire"),
        "装修及小型工程承包": ((".55", ".68", ".025", ".045"), (".035", ".060"), "renovation_materials_and_separate_specialist_trades"),
        "工程项目承包": ((".64", ".75", ".025", ".045"), (".030", ".050"), "bulk_project_materials_and_separately_contracted_specialist_trades"),
        "综合工程承包": ((".65", ".76", ".025", ".045"), (".030", ".055"), "construction_materials_and_separately_contracted_project_packages"),
        "专业工程承包": ((".50", ".64", ".025", ".045"), (".035", ".060"), "retained_installation_materials_and_selected_specialist_packages"),
        "单店经营": ((".32", ".43", ".055", ".085"), (".045", ".070"), "single_outlet_menu_inputs_and_premises"),
        "城配配送": ((".38", ".49", ".045", ".065"), (".035", ".055"), "own_delivery_staff_with_fuel_tolls_and_route_subcontracts"),
        "专线运输": ((".35", ".49", ".040", ".060"), (".035", ".060"), "own_route_vehicles_fuel_tolls_and_purchased_supplementary_capacity"),
        "仓储+运输": ((".27", ".42", ".045", ".065"), (".035", ".060"), "warehouse_consumables_vehicle_inputs_and_external_space_services"),
        "网络化综合物流": ((".58", ".72", ".040", ".060"), (".040", ".065"), "external_linehaul_and_peak_capacity_separate_from_own_network_labor"),
        "合同物流": ((".35", ".48", ".045", ".065"), (".035", ".060"), "contract_vehicle_inputs_and_purchased_supplementary_routes"),
        "多门店经营": ((".36", ".47", ".045", ".070"), (".040", ".065"), "food_ingredients_packaging_and_external_semifinished_inputs"),
        "社区门店经营": ((".33", ".44", ".055", ".080"), (".040", ".065"), "community_menu_ingredients_packaging_and_outside_inputs"),
        "酒店/餐饮综合经营": ((".24", ".37", ".060", ".090"), (".045", ".075"), "food_beverage_laundry_supplies_and_external_event_vendors"),
        "连锁门店经营": ((".37", ".48", ".045", ".070"), (".040", ".065"), "chain_food_ingredients_packaging_and_external_semifinished_inputs"),
        "技术咨询": ((".08", ".17", ".030", ".050"), (".035", ".060"), "external_test_resources_data_and_selected_specialist_reviews"),
        "研发服务项目制": ((".23", ".37", ".045", ".070"), (".040", ".065"), "research_consumables_trial_parts_and_outside_test_resources"),
        "检测/技术服务": ((".13", ".24", ".045", ".070"), (".035", ".060"), "sampling_consumables_and_external_specialist_reviews_not_owned_equipment"),
        "综合研发与工程技术服务": ((".24", ".38", ".055", ".080"), (".040", ".065"), "external_research_materials_specialist_processing_and_test_resources"),
        "专业技术项目服务": ((".24", ".36", ".045", ".070"), (".040", ".065"), "external_technical_components_and_specialist_resources_before_internal_transition"),
    }
    if model not in plans:
        raise ValueError(f"业务模式{model!r}缺少独立外采与自有人员归属预算，不能使用行业总成本代替")
    return plans[model]


def capital_update_budget(config, number, month, prototype_assets):
    local = rng(config["seed"], "capital-plan-v102", number, month)
    return decimal(money(prototype_assets * fraction(local, ".0020", ".0035")))


def workload_semantics(profile, industry):
    if profile["business_model"] == "SaaS订阅服务":
        return "normalized_account_service_months"
    if profile["business_model"] == "企业外包服务":
        return "accepted_customer_service_seat_hours"
    return {"制造": "accepted_manufacturing_piece_or_batch_equivalents",
            "批发零售": "net_sold_goods_bundle_equivalents", "信息技术": "accepted_software_deliverables_or_license_service_months",
            "商务服务": "accepted_advisory_hours_or_project_deliverable_packages", "建筑": "accepted_construction_quantity_equivalents",
            "物流运输": "completed_delivery_or_transport_trip_equivalents", "餐饮住宿": "meal_or_guest_room_service_equivalents",
            "科研技术": "accepted_test_report_or_technical_service_hour_equivalents"}[industry]


def fraction(random, low, high):
    quantum = Decimal(1000000)
    return Decimal(random.randint(int(decimal(low) * quantum), int(decimal(high) * quantum))) / quantum


def _d(value, places=".000001"):
    return decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def _months(first, last):
    cursor, end = date.fromisoformat(first).replace(day=1), date.fromisoformat(last)
    while cursor <= end:
        yield cursor
        cursor = date(cursor.year + int(cursor.month == 12), cursor.month % 12 + 1, 1)


def feature_plan(number, variant_for):
    """Choose mechanisms without looking at generated profits or balances."""
    family = (number - 1) % 40 + 1
    features = []
    if family in EQUAL_FAMILIES:
        features.append("equal_real_months")
    if family in FAMILIES:
        features.extend(("ratio_comparison", "lawful_revision"))
        if variant_for(number, "B2") != "counterexample":
            features.append("confirmed_zero_tax")
        else:
            features.append("confirmed_positive_tax")
        features.append({"typical": "two_month_loss_then_recovery",
                         "transfer": "three_month_continuous_loss",
                         "counterexample": "isolated_loss_without_continuity"}[variant_for(number, "C2")])
    elif number % 10 in (1, 4, 7):
        features.append(("continuous_loss", "revenue_contraction", "social_coverage_gap", "business_exit_and_asset_review")[number % 4])
    return features


def contract_interest_rate(config, enterprise_id, event_type, object_ref):
    """Contract prices are selected before profit calculations."""
    local = rng(config["seed"], "contract-interest-price-v102", enterprise_id, event_type, object_ref)
    low, high = (".038", ".065") if event_type == "银行借款" else (".032", ".048")
    return fraction(local, low, high)


def financing_costs(config, enterprise_id, period_start, period_end, ledgers, operations):
    """Calculate actual/365 monthly interest from dated principal checkpoints.

    A repayment changes principal on its documented checkpoint day. Interest
    is settled at each month end in this model, so no unpaid-interest balance
    is silently assumed to be zero.
    """
    first, last = date.fromisoformat(period_start), date.fromisoformat(period_end)
    loan_terms, bond_terms = [], []
    for ledger in ledgers:
        rate = contract_interest_rate(config, enterprise_id, "银行借款", ledger["business_object_ref"])
        segments = []
        day = first
        while day <= last:
            applicable = [point for point in ledger["checkpoints"] if point["balance_as_of_date"] <= day.isoformat()]
            principal = decimal(applicable[-1]["outstanding_principal"]) if applicable else Decimal(0)
            if segments and decimal(segments[-1]["principal"]) == principal:
                segments[-1]["days"] += 1
                segments[-1]["period_end"] = day.isoformat()
            else:
                segments.append({"period_start": day.isoformat(), "period_end": day.isoformat(), "principal": money(principal), "days": 1})
            day += timedelta(days=1)
        principal_days = sum((decimal(segment["principal"]) * segment["days"] for segment in segments), Decimal(0))
        if principal_days:
            loan_terms.append({"business_object_ref": ledger["business_object_ref"], "annual_interest_rate": rate,
                               "day_count_basis": 365, "interest_amount": money(principal_days * rate / Decimal(365)),
                               "daily_principal_days": segments, "payment_status": "settled_at_period_end"})
    for row in operations:
        if row["event_type"] != "债券发行" or row["event_date"] > period_end:
            continue
        rate = contract_interest_rate(config, enterprise_id, "债券发行", row["business_object_ref"])
        accrual_start = max(period_start, row["event_date"])
        days = (last - date.fromisoformat(accrual_start)).days + 1
        principal = decimal(row["amount"])
        bond_terms.append({"business_object_ref": row["business_object_ref"], "annual_interest_rate": rate,
                           "day_count_basis": 365, "interest_amount": money(principal * days * rate / Decimal(365)),
                           "daily_principal_days": [{"period_start": accrual_start, "period_end": period_end, "principal": money(principal), "days": days}],
                           "payment_status": "settled_at_period_end"})
    loan_cost = sum((decimal(term["interest_amount"]) for term in loan_terms), Decimal(0))
    bond_cost = sum((decimal(term["interest_amount"]) for term in bond_terms), Decimal(0))
    return loan_cost, bond_cost, loan_terms, bond_terms


def initial_reporting_estimate(config, final_process, monthly_record_id):
    """An original report estimates only documented acceptance quantity.

    Customer acceptance documents arrive in stages. Labor, fixed spending and
    contract interest are already booked and are not scaled with the initial
    sales estimate. The later revision uses completed acceptance documents.
    """
    local = rng(config["seed"], "lawful-initial-acceptance-coverage", final_process["enterprise_id"], final_process["source_record_ref"])
    coverage = fraction(local, ".91", ".97")
    drivers = deepcopy(final_process["drivers"])
    delivered_units = _d(decimal(drivers["delivered_units"]) * coverage)
    drivers["delivered_units"] = drivers["material_units"] = delivered_units
    drivers["initial_acceptance_coverage_rate"] = coverage
    revenue = decimal(money(delivered_units * decimal(drivers["unit_price"])))
    direct = decimal(money(delivered_units * decimal(drivers["material_unit_cost"]) +
                           decimal(drivers["rework_units"]) * decimal(drivers["rework_unit_cost"])))
    payroll, finance = decimal(final_process["payroll_expense"]), decimal(final_process["finance_expense"])
    overhead = decimal(money(decimal(drivers["base_overhead"]) * decimal(drivers["coverage_fraction"]) +
                             revenue * decimal(drivers["variable_overhead_rate"]) +
                             sum(decimal(drivers[key]) for key in ("capacity_reservation_cost", "asset_impairment_expense", "depreciation_expense"))))
    profit = revenue - direct - payroll - overhead - finance
    income_tax = decimal(money(profit * decimal(drivers["income_tax_rate"]))) if profit > 0 else Decimal(0)
    values = {"operating_revenue": money(revenue), "main_business_revenue": money(revenue * decimal(drivers["main_business_share"])),
              "total_profit": money(profit), "net_profit": money(profit - income_tax)}
    estimate = {
        "enterprise_id": final_process["enterprise_id"], "stat_month": final_process["stat_month"],
        "source_system": final_process["source_system"], "source_record_ref": final_process["source_record_ref"],
        "monthly_record_id": monthly_record_id, "report_revision": 1,
        "record_ref": {"table": "enterprise_operation_monthly", "primary_key": {"monthly_record_id": monthly_record_id}},
        "period_start": final_process["period_start"], "period_end": final_process["period_end"],
        "revenue": money(revenue), "direct_cost": money(direct), "payroll_expense": money(payroll),
        "operating_expense": money(overhead), "finance_expense": money(finance), "other_income": "0.00",
        "total_profit": money(profit), "income_tax_expense": money(income_tax), "net_profit": money(profit - income_tax),
        "tax_payment": final_process["tax_payment"], "reported_employer_social_cost": final_process["reported_employer_social_cost"],
        "other_labor_cost": final_process["other_labor_cost"], "drivers": drivers,
        "invoice_and_payment_timing_reference": final_process["invoice_and_payment_timing_reference"],
        "timing_reference_kind": final_process["timing_reference_kind"], "opening_prototype": final_process["opening_prototype"],
        "estimate_kind": "lawful_initial_acceptance_estimate", "driver_selection_stage": "acceptance_document_estimate_before_reported_profit",
        "mechanism": "初报按已收到验收资料估计计价工作量及对应耗用，工资、固定场地费用和合同利息按已发生数保留；变动支出及利润逐项重新计算。补齐客户验收资料后，在同一统计月发布修订；实际开票及税款支付没有因初报估计而改变。",
    }
    return values, estimate


def opening_stocks(config, source_profile, basic, number, features):
    """Use the frozen business scale as an explicit opening estimate.

    The source totals are scale anchors, not observations copied to every
    snapshot. Early-stage enterprises start at registration; older enterprises
    use a window-opening estimate. Any pre-window distribution is chosen here,
    before future income or the eventual balance sheet is known.
    """
    local = rng(config["seed"], "opening-stock", number)
    assets = decimal(source_profile["total_assets"])
    liabilities = decimal(source_profile["total_liabilities"])
    if number > 40:
        assets = decimal(money(assets * fraction(local, ".74", "1.36")))
        liabilities = decimal(money(liabilities * fraction(local, ".72", "1.22")))
    if assets <= liabilities:
        raise ValueError(f"{basic['enterprise_id']}: opening prototype has no positive equity for this funding plan")
    established = date.fromisoformat(basic["established_date"])
    first = date.fromisoformat(config["window_start"])
    start = max(first - timedelta(days=1), established)
    business_age_days = (start - established).days
    composition = rng(config["seed"], "opening-equity-composition-budget", number, basic["industry"])
    if business_age_days < 365:
        retained_fraction = _d(fraction(composition, ".02", ".08") * Decimal(business_age_days) / Decimal(365))
    elif business_age_days < 365 * 3:
        retained_fraction = fraction(composition, ".20", ".40")
    else:
        retained_fraction = fraction(composition, ".35", ".65")
    retained_budget = decimal(money((assets - liabilities) * retained_fraction))
    contributed_equity_budget = assets - liabilities - retained_budget
    risk = "business_exit_and_asset_review" in features
    cash_share = Decimal(".70") if risk else fraction(local, ".32", ".40")
    receivable_share = (Decimal(".12") if risk else
                        fraction(local, ".07", ".13") if basic["industry"] in ("批发零售", "餐饮住宿") else
                        fraction(local, ".16", ".22"))
    inventory_share = (Decimal(".07") if risk else
                       fraction(local, ".20", ".26") if basic["industry"] == "批发零售" else
                       fraction(local, ".025", ".055") if basic["industry"] in ("信息技术", "商务服务", "科研技术") else
                       fraction(local, ".10", ".15"))
    cash, receivables, inventory = [decimal(money(assets * share)) for share in (cash_share, receivable_share, inventory_share)]
    fixed = assets - cash - receivables - inventory
    distribution_ratio = fraction(rng(config["seed"], "pre-window-profit-distribution-plan", number), ".45", ".65") if risk and business_age_days >= 365 else Decimal(0)
    distribution = decimal(money(retained_budget * distribution_ratio))
    cash -= distribution
    if cash < 0 or fixed < 0:
        raise ValueError(f"{basic['enterprise_id']}: opening stock composition cannot finance its preselected distribution")
    payables = decimal(money(liabilities * fraction(local, ".64", ".75")))
    other = liabilities - payables
    return {
        "opening_stock_date": start.isoformat(),
        "opening_stock_kind": ("registration_opening_funding_estimate" if established >= first else "window_opening_prototype_estimate"),
        "prototype_assets_before_distribution": money(assets),
        "prototype_liabilities": money(liabilities),
        "opening_business_age_days": business_age_days,
        "pre_window_distribution_age_eligible": business_age_days >= 365,
        "opening_equity_composition_scope": "preselected_budget_estimate_not_observed_history",
        "opening_owner_contributed_equity_budget": money(contributed_equity_budget),
        "opening_retained_earnings_budget": money(retained_budget),
        "opening_retained_earnings_budget_fraction": retained_fraction,
        "opening_retained_earnings_after_distribution": money(retained_budget - distribution),
        "pre_window_distribution_ratio": distribution_ratio,
        "pre_window_owner_distribution": money(distribution),
        "pre_window_owner_distribution_date": start.isoformat() if distribution else None,
        "pre_window_distribution_reason": ("从窗口前独立预估的留存收益预算中按预选比例执行分配，不将资本性投入当作经营利润。" if distribution else
                                           "开业不足一年保留所有者投入支持营运，不安排窗口前分配。" if business_age_days < 365 else
                                           "未安排额外窗口前股东分配。"),
        "opening_cash": money(cash), "opening_trade_receivables": money(receivables),
        "opening_inventory": money(inventory), "opening_fixed_assets": money(fixed),
        "opening_trade_payables": money(payables), "opening_other_liabilities": money(other),
        "opening_owners_equity": money(assets - distribution - liabilities),
        "opening_total_assets": money(assets - distribution), "opening_total_liabilities": money(liabilities),
    }


def operating_months(config, source_months, source_profile, basic, profile, number, variant_for, row_factory, metadata, ledgers, operations):
    """Generate one chronological sequence, with each reference finalized first."""
    random = rng(config["seed"], "economic-parameters-v102", number)
    settings = config.get("generation", {})
    revenue_scale = (Decimal(1) if number <= 40 else fraction(random, settings.get("new_enterprise_scale_min", ".70"), settings.get("new_enterprise_scale_max", "1.40")))
    employee_scale = Decimal(1) if number <= 40 else fraction(random, ".78", "1.24")
    trend = profile["operating_trend"]
    annual_growth = (fraction(random, ".12", ".30") if "增长" in trend else
                     fraction(random, "-.25", "-.08") if "下降" in trend else fraction(random, "-.04", ".06"))
    templates = sorted(source_months, key=lambda row: row["stat_month"])
    average = sum(decimal(row["operating_revenue"]) for row in templates) / Decimal(len(templates))
    average_staff = Decimal(sum(row["employee_count"] for row in templates)) / Decimal(len(templates))
    social_per_person = _d(sum(decimal(row["social_security_amount"]) for row in templates) /
                           Decimal(sum(row["social_security_count"] for row in templates)))
    # A synthetic social-payment share of comprehensive employer labor cost,
    # not a gross-wage contribution rate or a statutory policy parameter.
    employer_social_rate = Decimal(".23")
    job_pay_factor = fraction(rng(config["seed"], "job-pay", number), ".96", "1.16")
    monthly_salary = _d(social_per_person / employer_social_rate * job_pay_factor)
    cost_ranges, variable_range, cost_plan_kind = cost_plan(profile, basic["industry"])
    material_low, material_high, fixed_low, fixed_high = cost_ranges
    base_material_rate = fraction(random, material_low, material_high)
    base_overhead = _d(average * fraction(random, fixed_low, fixed_high))
    variable_overhead_rate = fraction(random, *variable_range)
    utility_share = fraction(rng(config["seed"], "facility-energy-budget-allocation", number, profile["business_model"]), ".15", ".30")
    travel_share = fraction(rng(config["seed"], "business-travel-budget-allocation", number, profile["business_model"]), ".08", ".18")
    unit_count = Decimal(random.randint(80, 700))
    unit_price = decimal(money(average / unit_count))
    features = feature_plan(number, variant_for)
    opening = opening_stocks(config, source_profile, basic, number, features)
    depreciation_budget = decimal(money(decimal(opening["opening_fixed_assets"]) * Decimal(".005")))
    impairment_budget = decimal(money(decimal(opening["opening_fixed_assets"]) * Decimal(".72"))) if "business_exit_and_asset_review" in features else Decimal(0)
    first_revenue, last_revenue = decimal(templates[0]["operating_revenue"]), decimal(templates[-1]["operating_revenue"])
    seasonal_levels = {}
    for position, template in enumerate(templates):
        linear = first_revenue + (last_revenue - first_revenue) * Decimal(position) / Decimal(len(templates) - 1)
        denominator = linear if "增长" in trend or "下降" in trend else average
        seasonal_levels[date.fromisoformat(template["stat_month"]).month] = decimal(template["operating_revenue"]) / denominator
    first = date.fromisoformat(config["window_start"])
    established = date.fromisoformat(basic["established_date"])
    previous_revenue = Decimal(0) if established >= first else decimal(money(average * revenue_scale))
    normal_input_budget = _d(average * revenue_scale * base_material_rate)
    inventory_buffer_months = fraction(rng(config["seed"], "inventory-buffer", number), ".10", ".20")
    previous_inventory = decimal(opening["opening_inventory"])
    opening_sales_pending_rate = fraction(rng(config["seed"], "opening-sales-invoice-pending", number), ".10", ".30")
    opening_purchase_pending_rate = fraction(rng(config["seed"], "opening-purchase-invoice-pending", number), ".10", ".30")
    sales_pending = Decimal(0) if established >= first else decimal(money(previous_revenue * opening_sales_pending_rate))
    opening_procurement_basis = (Decimal(0) if established >= first else decimal(money(
        normal_input_budget + base_overhead + previous_revenue * variable_overhead_rate +
        capital_update_budget(config, number, config["window_start"], decimal(opening["prototype_assets_before_distribution"])))) )
    purchase_pending = Decimal(0) if established >= first else decimal(money(opening_procurement_basis * opening_purchase_pending_rate))
    opening_utility_basis = Decimal(0) if established >= first else decimal(money(base_overhead * utility_share))
    opening_travel_basis = Decimal(0) if established >= first else decimal(money(previous_revenue * variable_overhead_rate * travel_share))
    utility_pending = decimal(money(opening_utility_basis * opening_purchase_pending_rate))
    travel_pending = decimal(money(opening_travel_basis * opening_purchase_pending_rate))
    results, processes = [], []
    family = (number - 1) % 40 + 1
    variation_limit = decimal(settings.get("normal_monthly_variation", ".05"))
    for month in _months(config["window_start"], config["window_end"]):
        end = month_end(month)
        if end < established:
            continue
        period_start = max(month, established)
        period_fraction = Decimal((end - period_start).days + 1) / Decimal((end - month).days + 1)
        local = rng(config["seed"], "operating-month-v102", number, month.isoformat())
        elapsed = (month.year - first.year) * 12 + month.month - first.month
        strength = fraction(rng(config["seed"], "seasonality", number), ".82", ".98")
        seasonal = seasonal_levels[month.month] if number <= 40 else Decimal(1) + (seasonal_levels[month.month] - Decimal(1)) * strength
        trend_factor = Decimal(1) + annual_growth * Decimal(elapsed - 12) / Decimal(12)
        level = (first_revenue if number <= 40 and ("增长" in trend or "下降" in trend) else average * revenue_scale)
        workload = level * seasonal * trend_factor * fraction(local, Decimal(1) - variation_limit, Decimal(1) + variation_limit)
        mechanisms = ["客户交付量、验收单价、原型季节性及经营趋势决定确认收入。"]
        material_rate = fraction(local, base_material_rate - Decimal(".015"), base_material_rate + Decimal(".015"))
        transition_factor = (Decimal(".82") if profile["business_model"] in ("系统集成与定制开发", "专业技术项目服务") and month >= date(2026, 3, 1) else
                             Decimal(".94") if profile["business_model"] in ("系统集成与定制开发", "专业技术项目服务") and month >= date(2025, 10, 1) else
                             Decimal(".88") if profile["business_model"] == "专业工程承包" and month >= date(2025, 12, 1) else Decimal(1))
        material_rate *= transition_factor
        if transition_factor != 1:
            mechanisms.append("按此前确定的产品组合和内部组件研发安排，减少被退出项目的外部供货或专项外协，保留自有人员预算。")
        if family in EQUAL_FAMILIES and month.isoformat() in ("2025-11-01", "2025-12-01"):
            # Two separately scheduled, equally priced contract lots. Neither
            # is copied from another generated observation.
            workload = average * revenue_scale * Decimal("1.04")
            mechanisms.append("按两次独立月度合同的相同计价工作量与固定单价完成验收；各月工资、耗用和结算独立发生。")
        if family in FAMILIES and month.isoformat() in ("2026-04-01", "2026-05-01", "2026-06-01"):
            workload = average * revenue_scale * Decimal("1.10")
        if family in FAMILIES and month.isoformat() in ("2026-06-01", "2026-07-01"):
            workload = average * revenue_scale * (Decimal(".99") if month.month == 6 else Decimal("1.72"))
            material_rate = base_material_rate * (Decimal(".77") if month.month == 6 else Decimal("1.08"))
            mechanisms.append("小批次存量客户服务与大批次集中交付采用不同产品组合和外包耗用系数。")
        if "revenue_contraction" in features and month.isoformat() in ("2026-04-01", "2026-05-01", "2026-06-01"):
            workload *= Decimal(".57")
            mechanisms.append("客户延后部分验收，工资合同及场地支出仍按既定安排发生。")
        if "business_exit_and_asset_review" in features:
            workload *= Decimal(".50")
            mechanisms.append("退出非核心业务后交付量降低，既定最低产能和长期场地承诺继续支付。")
        delivered_units = _d(workload / unit_price * period_fraction)
        revenue = decimal(money(delivered_units * unit_price))
        staff_trend = Decimal(1) + annual_growth * Decimal(max(elapsed - 4, 0) - 14) / Decimal(30)
        staffing_reference_revenue = _d(average * revenue_scale)
        lagged_revenue = decimal(results[-3]["operating_revenue"]) if len(results) >= 3 else decimal(money(staffing_reference_revenue))
        lagged_capacity_factor = Decimal(".92") + Decimal(".08") * lagged_revenue / staffing_reference_revenue
        staff = int((average_staff * employee_scale * staff_trend * lagged_capacity_factor).to_integral_value(rounding=ROUND_HALF_UP))
        staff += local.choice((-1, 0, 0, 0, 1)) if staff > 5 else 0
        if staff < 1 or not Decimal(0) < material_rate < Decimal(1):
            raise ValueError(f"{basic['enterprise_id']} {month}: capacity/material parameter conflict")
        social_count = staff - local.choice((0, 0, 0, 1)) if staff > 1 else staff
        if "social_coverage_gap" in features and month.isoformat() in ("2026-04-01", "2026-05-01", "2026-06-01"):
            social_count = staff * 3 // 4
            mechanisms.append("参保报送按本单位申报范围记录，月末人员还包含其他用工安排；差异不足以独立确认违规。")
        shock = False
        if family in FAMILIES:
            role = variant_for(number, "C2")
            shock_months = {"typical": (4, 5), "transfer": (4, 5, 6), "counterexample": (4, 6)}[role]
            shock = month.year == 2026 and month.month in shock_months
        if "continuous_loss" in features:
            shock = month.year == 2026 and month.month in (4, 5, 6)
        material_unit_cost = decimal(money(unit_price * material_rate))
        if basic["industry"] == "批发零售":
            rework_kind = "affected_goods_batch_repurchase_reverse_logistics_and_inspection"
            rework_quantity_fraction = fraction(rng(config["seed"], "affected-trade-batch-plan", number, profile["business_model"]), ".25", ".45")
            repurchase_price_factor = fraction(rng(config["seed"], "replacement-goods-purchase-quote", number, profile["business_model"]), "1.01", "1.08")
            reverse_logistics_quote_rate = fraction(rng(config["seed"], "reverse-logistics-inspection-quote", number, profile["business_model"]), ".025", ".045")
            rework_quote_rate = base_material_rate * repurchase_price_factor + reverse_logistics_quote_rate
            reserved_capacity_rate = fraction(rng(config["seed"], "return-batch-handling-capacity", number, profile["business_model"]), ".035", ".055")
        else:
            rework_kind = "external_specialist_remediation_or_urgent_redelivery"
            rework_quantity_fraction = Decimal(".95")
            repurchase_price_factor = reverse_logistics_quote_rate = Decimal(0)
            rework_quote_rate = fraction(rng(config["seed"], "exceptional-remediation-price", number, profile["business_model"]), ".95", "1.15")
            reserved_capacity_rate = Decimal(".12")
        rework_units = _d(staffing_reference_revenue / unit_price * rework_quantity_fraction * period_fraction) if shock else Decimal(0)
        rework_unit_cost = decimal(money(unit_price * rework_quote_rate))
        capacity_cost = decimal(money(staffing_reference_revenue * reserved_capacity_rate * period_fraction)) if shock else Decimal(0)
        if "business_exit_and_asset_review" in features:
            capacity_cost += decimal(money(average * Decimal(".055") * period_fraction))
        if shock:
            if basic["industry"] == "批发零售":
                mechanisms.append("先前交付的部分商品批次按质量处置协议无偿替换，旧货无法重新销售且不假定供应商退款；替换合格商品按采购报价重购，另计逆向物流和复检费用，约定的退换货处理场地及服务能力分别预算。正常销售与售后替换分开确认，替换不增加销售收入。")
            else:
                mechanisms.append("既有批次返工或项目重交付不形成新增销售；额外耗用和保留的临时产能按预定工作量计费。")
            if profile["business_model"] == "城配配送":
                mechanisms.append("路线调整后的重派采用高峰临时外包直达报价，与常规油耗、路桥和分段外包成本分别计价。")
        depreciation = decimal(money(depreciation_budget * period_fraction))
        impairment = impairment_budget if "business_exit_and_asset_review" in features and month.isoformat() == "2026-05-01" else Decimal(0)
        direct = decimal(money(delivered_units * material_unit_cost + rework_units * rework_unit_cost))
        payroll = decimal(money(Decimal(staff) * monthly_salary * period_fraction))
        overhead = decimal(money(base_overhead * period_fraction + revenue * variable_overhead_rate + capacity_cost + depreciation + impairment))
        finance_rate = Decimal(".003")
        settlement_fee = decimal(money(previous_revenue * finance_rate * period_fraction))
        loan_interest, bond_interest, loan_terms, bond_terms = financing_costs(
            config, basic["enterprise_id"], period_start.isoformat(), end.isoformat(), ledgers, operations)
        finance = settlement_fee + loan_interest + bond_interest
        profit = revenue - direct - payroll - overhead - finance
        income_tax_rate = Decimal(".20")
        income_tax = decimal(money(profit * income_tax_rate)) if profit > 0 else Decimal(0)
        main_share = fraction(local, ".88", ".985")
        sales_current, purchase_current = fraction(local, ".65", ".92"), fraction(local, ".55", ".90")
        sales_issued = decimal(money(revenue * sales_current))
        sales_amount = sales_issued + sales_pending
        sales_closing_pending = revenue - sales_issued
        inventory_target_factor = Decimal(1) + annual_growth * Decimal(elapsed) / Decimal(48)
        closing_inventory = decimal(money(decimal(opening["opening_inventory"]) + normal_input_budget * inventory_buffer_months * inventory_target_factor))
        inventory_change = closing_inventory - previous_inventory
        capex = capital_update_budget(config, number, month.isoformat(), decimal(opening["prototype_assets_before_distribution"]))
        purchased_period_expense = overhead - depreciation - impairment
        procurement_basis = direct + purchased_period_expense + capex + inventory_change
        if procurement_basis < 0:
            raise ValueError(f"{basic['enterprise_id']} {month}: consumed inputs and preselected inventory plan imply negative purchases {money(procurement_basis)}")
        purchased_issued = decimal(money(procurement_basis * purchase_current))
        purchase = purchased_issued + purchase_pending
        purchase_closing_pending = procurement_basis - purchased_issued
        utility_basis = decimal(money(base_overhead * period_fraction * utility_share))
        travel_basis = decimal(money(revenue * variable_overhead_rate * travel_share))
        other_period_basis = purchased_period_expense - utility_basis - travel_basis
        if other_period_basis < 0:
            raise ValueError(f"{basic['enterprise_id']} {month}: energy/travel budget allocation exceeds purchased period expense")
        utility_issued = decimal(money(utility_basis * purchase_current))
        travel_issued = decimal(money(travel_basis * purchase_current))
        utility_invoice, travel_invoice = utility_issued + utility_pending, travel_issued + travel_pending
        utility_closing, travel_closing = utility_basis - utility_issued, travel_basis - travel_issued
        prior_accrual = sum((decimal(process["income_tax_expense"]) for process in processes), Decimal(0))
        prior_paid = sum((decimal(process["tax_payment"]) for process in processes), Decimal(0))
        due_tax = prior_accrual - prior_paid
        if due_tax < 0:
            raise ValueError(f"{basic['enterprise_id']} {month}: tax payment notices and actual payments disagree")
        exempt = "confirmed_zero_tax" in features and month.isoformat() == "2026-02-01"
        tax_payment = Decimal(0) if exempt else due_tax
        if exempt:
            mechanisms.append("申报缴款计划中该月无现金税款支付；纳税义务及损益税费不据此归零。")
        timing_kind = ("previous_canonical_month" if results else "no_pre_establishment_activity" if established >= first else "opening_prototype")
        values = {
            "employee_count": staff, "social_security_count": social_count,
            "social_security_amount": money(Decimal(social_count) * social_per_person * period_fraction),
            "tax_amount": money(tax_payment),
            "invoice_purchase_amount": money(purchase),
            "invoice_sales_amount": money(sales_amount),
            "utility_invoice_amount": money(utility_invoice), "travel_hospitality_invoice_amount": money(travel_invoice),
            "operating_revenue": money(revenue), "main_business_revenue": money(revenue * main_share),
            "total_profit": money(profit), "net_profit": money(profit - income_tax),
            "monthly_record_id": f"MR-{number:06d}-{month:%Y%m}-R1", "enterprise_id": basic["enterprise_id"],
            "stat_month": month.isoformat(), "period_start": period_start.isoformat(), "period_end": end.isoformat(), "amount_unit": "元",
        }
        row = row_factory("enterprise_operation_monthly", **values)
        row.update(metadata(config, "enterprise_operation_monthly", f"MONTH-{basic['enterprise_id']}-{month:%Y%m}", end))
        drivers = {
            "delivered_units": delivered_units, "unit_price": unit_price, "material_units": delivered_units,
            "delivery_unit_kind": workload_semantics(profile, basic["industry"]),
            "direct_cost_basis": "delivered_materials_goods_and_subcontract_services",
            "business_model_cost_plan": cost_plan_kind, "rework_quote_rate": rework_quote_rate,
            "base_material_rate": base_material_rate, "reference_monthly_revenue_budget": staffing_reference_revenue,
            "rework_mechanism_kind": rework_kind, "rework_quantity_fraction": rework_quantity_fraction,
            "replacement_purchase_price_factor": repurchase_price_factor,
            "reverse_logistics_inspection_quote_rate": reverse_logistics_quote_rate,
            "reserved_remediation_capacity_rate": reserved_capacity_rate,
            "outsourcing_transition_factor": transition_factor,
            "material_unit_cost": material_unit_cost, "staff": staff, "monthly_salary": monthly_salary,
            "salary_per_person": monthly_salary, "employer_social_rate": employer_social_rate, "job_pay_factor": job_pay_factor,
            "social_rate_semantics": "synthetic_payroll_reporting_budget_coefficient_not_statutory_rate",
            "payroll_cost_basis": "comprehensive_employer_labor_cost_including_reported_social_contribution",
            "tax_payment_semantics": "synthetic_income_tax_prepayment_schedule",
            "staffing_lag_months": 3, "lagged_capacity_reference_revenue": money(lagged_revenue), "lagged_capacity_factor": lagged_capacity_factor,
            "staffing_reference_revenue": staffing_reference_revenue,
            "staffing_reference_kind": "three_month_lagged_canonical_revenue" if len(results) >= 3 else "opening_staffing_plan_prototype",
            "staffing_reference_month": results[-3]["stat_month"] if len(results) >= 3 else None,
            "coverage_fraction": period_fraction, "base_overhead": base_overhead, "variable_overhead_rate": variable_overhead_rate,
            "rework_units": rework_units, "rework_unit_cost": rework_unit_cost, "capacity_reservation_cost": capacity_cost,
            "asset_impairment_expense": impairment, "depreciation_expense": depreciation, "finance_rate": finance_rate,
            "finance_expense_kind": "settlement_processing_fee_and_contract_interest", "finance_billing_base": "previous_finalized_business_month_revenue",
            "loan_interest_expense": money(loan_interest), "bond_interest_expense": money(bond_interest),
            "loan_interest_terms": loan_terms, "bond_interest_terms": bond_terms,
            "invoice_current_weight": sales_current, "invoice_previous_weight": "0",
            "purchase_current_weight": purchase_current, "purchase_previous_weight": "0",
            "invoice_process_mode": "conserved_current_and_prior_pending_queues",
            "sales_invoice_opening_pending": money(sales_pending), "sales_invoice_current_issued": money(sales_issued),
            "sales_invoice_closing_pending": money(sales_closing_pending),
            "procurement_consumed_direct_cost": money(direct), "procurement_purchased_period_expense": money(purchased_period_expense),
            "capital_expenditure": money(capex), "procurement_basis": money(procurement_basis),
            "purchase_invoice_opening_pending": money(purchase_pending), "purchase_invoice_current_issued": money(purchased_issued),
            "purchase_invoice_closing_pending": money(purchase_closing_pending),
            "utility_expense_base": money(utility_basis), "travel_hospitality_expense_base": money(travel_basis),
            "other_purchased_period_expense_base": money(other_period_basis),
            "utility_invoice_opening_pending": money(utility_pending), "utility_invoice_current_issued": money(utility_issued),
            "utility_invoice_closing_pending": money(utility_closing),
            "travel_hospitality_invoice_opening_pending": money(travel_pending), "travel_hospitality_invoice_current_issued": money(travel_issued),
            "travel_hospitality_invoice_closing_pending": money(travel_closing),
            "utility_opening_prototype_base": money(opening_utility_basis), "travel_opening_prototype_base": money(opening_travel_basis),
            "opening_inventory": opening["opening_inventory"], "previous_inventory": money(previous_inventory),
            "normal_monthly_input_budget": normal_input_budget, "inventory_buffer_months": inventory_buffer_months,
            "inventory_target_factor": inventory_target_factor, "closing_inventory": money(closing_inventory), "inventory_change": money(inventory_change),
            "invoice_opening_reference_kind": timing_kind,
            "opening_sales_pending_rate": opening_sales_pending_rate, "opening_purchase_pending_rate": opening_purchase_pending_rate,
            "opening_procurement_basis_estimate": money(opening_procurement_basis),
            "vat_settlement_assumption": "same_period_pass_through_net_zero_no_vat_timing_assets_or_liabilities",
            "sales_tax_payment_rate": "0", "income_tax_rate": income_tax_rate, "tax_payment_exempt": exempt,
            "tax_payment_basis": "prior_known_income_tax_accrual_after_paid_credits",
            "prior_cumulative_income_tax_expense": money(prior_accrual), "prior_cumulative_tax_paid": money(prior_paid),
            "scheduled_tax_payment": money(due_tax),
            "main_business_share": main_share, "utility_purchase_share": utility_share, "travel_purchase_share": travel_share,
            "social_count": social_count, "social_per_person": social_per_person,
        }
        process = {
            "enterprise_id": basic["enterprise_id"], "stat_month": month.isoformat(), "source_system": row["source_system"],
            "source_record_ref": row["source_record_ref"], "period_start": period_start.isoformat(), "period_end": end.isoformat(),
            "revenue": money(revenue), "direct_cost": money(direct), "payroll_expense": money(payroll),
            "operating_expense": money(overhead), "finance_expense": money(finance), "other_income": "0.00",
            "total_profit": money(profit), "income_tax_expense": money(income_tax), "net_profit": money(profit - income_tax),
            "tax_payment": row["tax_amount"], "reported_employer_social_cost": row["social_security_amount"],
            "other_labor_cost": money(payroll - decimal(row["social_security_amount"])),
            "invoice_and_payment_timing_reference": money(previous_revenue), "timing_reference_kind": timing_kind,
            "opening_prototype": timing_kind == "opening_prototype", "drivers": drivers,
            "driver_selection_stage": "before_revenue_and_profit", "business_features": list(features),
            "mechanism": "".join(mechanisms) + "外采材料、商品和所选外包与自有岗位工作分开预算；人员依原型岗位及滞后招聘安排，综合用工成本由参保人均金额、合成申报预算系数与岗位成本参数形成，已包含报送的雇主社保支出，其他人工成本另列；利润由耗用、综合用工和具名费用逐项形成。财务费用包括按前月确认收入计收的结算手续费以及依独立合同年利率、实际本金余额时点和actual/365逐日计提的借款及债券利息；该模型合同均约定月末现金结息。发票以本期净价确认额或实际采购基数按预选比例开具，加上前期待开金额，余额继续留到下一期，期初加发生额减期末待开等于累计开票；采购以已耗用外采、库存净增加、可开票期间外采和资本更新预算形成，不包括工资、折旧或减值。所得税预缴现金按以前月份已计提且尚未支付的税费通知执行，暂缓月份的应缴在后续付款计划中继续结算，亏损且无以前应缴税费时不依收入持续预缴。税费与实付差额在资金台账列为应缴税负债或预缴税资产。增值税代收代付按同期间净额零的简化资金假设处理，不生成增值税时差余额或实际计缴明细。",
        }
        if not results:
            process["mechanism"] += ("成立当月的前期经营引用为零。" if established >= first else "首月引用为独立窗口前原型估计，不是已观察月报。")
        if impairment:
            process["mechanism"] += "退出的设备经用途与可回收性复核后确认预先设定的资产减值，不改变现金支付。"
        results.append(row)
        processes.append(process)
        previous_revenue = revenue
        previous_inventory = closing_inventory
        sales_pending, purchase_pending = sales_closing_pending, purchase_closing_pending
        utility_pending, travel_pending = utility_closing, travel_closing
    return results, processes, features, opening


def funding_schedule(config, basic, number, opening, months, features):
    """Fix named capital and distribution plans before any final stock total."""
    assets = decimal(opening["prototype_assets_before_distribution"])
    risk = "business_exit_and_asset_review" in features
    start = max(basic["established_date"], config["window_start"])
    schedules, distribution_decisions = [], []
    for row in months:
        when = row["period_end"]
        capex = capital_update_budget(config, number, row["stat_month"], assets)
        schedules.append({"flow_date": when, "flow_type": "capital_expenditure", "amount": money(capex), "reason": "依设备和系统维护预算分期更新，金额不依据最终资产余额倒算。"})
    if not risk:
        for when in ("2025-09-30", "2026-03-31"):
            if start <= when <= config["window_end"]:
                schedules.append({"flow_date": when, "flow_type": "owner_contribution", "amount": money(assets * Decimal(".025")), "reason": "经营年度开始及项目更新前，股东按事先预算补充自有营运资金。"})
        if start <= "2026-06-30":
            payout_fraction = fraction(rng(config["seed"], "annual-distribution-budget", number), ".08", ".14")
            planned = decimal(money(decimal(opening["opening_retained_earnings_after_distribution"]) * payout_fraction))
            known_profit = sum((decimal(row["net_profit"]) for row in months if row["period_end"] <= "2026-05-31"), Decimal(0))
            available = decimal(opening["opening_retained_earnings_after_distribution"]) + known_profit
            actual = planned if planned > 0 and available >= planned else Decimal(0)
            status = "executed" if actual > 0 else "deferred" if planned > 0 else "not_planned"
            reason = ("决议时点已结束月份的经营损益加期初未分配留存预算足以支持整笔既定分配，按计划执行。" if actual > 0 else
                      "决议时点可用留存不足以支持整笔计划，分配整体延期，保留资金用于经营。" if planned > 0 else
                      "期初未分配留存预算为零，不安排年度利润分配。")
            distribution_decisions.append({
                "planned_at": opening["opening_stock_date"], "decision_at": "2026-06-15",
                "knowledge_period_end": "2026-05-31", "planned_payment_date": "2026-06-30",
                "planned_budget_basis": "opening_remaining_retained_earnings_budget", "planned_payout_fraction": payout_fraction,
                "planned_amount": money(planned), "known_cumulative_net_profit": money(known_profit),
                "available_retained_earnings": money(available), "actual_amount": money(actual),
                "actual_payment_date": "2026-06-30" if actual > 0 else None, "status": status, "reason": reason,
            })
            if actual > 0:
                schedules.append({"flow_date": "2026-06-30", "flow_type": "owner_distribution", "amount": money(actual), "reason": reason})
    return {"schedule_scope": "preselected_business_funding_plan", "selection_stage": "before_snapshot_totals",
            "enterprise_id": basic["enterprise_id"], "opening": deepcopy(opening), "flows": schedules,
            "owner_distribution_decisions": distribution_decisions}


def snapshot_stocks(config, source_profile, basic, number, features, months, processes, ledgers, operations, opening, row_factory, metadata):
    """Build dated stock components using a cash and equity rollforward."""
    plan = funding_schedule(config, basic, number, opening, months, features)
    result, audit = [], []
    base_cash = decimal(opening["opening_cash"])
    base_receivables, base_inventory = decimal(opening["opening_trade_receivables"]), decimal(opening["opening_inventory"])
    base_fixed = decimal(opening["opening_fixed_assets"])
    base_payables, base_other = decimal(opening["opening_trade_payables"]), decimal(opening["opening_other_liabilities"])
    base_equity = decimal(opening["opening_owners_equity"])
    previous_retained = previous_owner = previous_financing = previous_distribution = None
    for text in config["profile_dates"]:
        applicable = [row for row in processes if row["period_end"] <= text]
        final_process = applicable[-1]
        revenue = decimal(final_process["revenue"])
        direct = decimal(final_process["direct_cost"])
        retained = sum((decimal(row["net_profit"]) for row in applicable), Decimal(0))
        depreciation = sum((decimal(row["drivers"]["depreciation_expense"]) for row in applicable), Decimal(0))
        impairment = sum((decimal(row["drivers"]["asset_impairment_expense"]) for row in applicable), Decimal(0))
        tax_accrual = sum((decimal(row["income_tax_expense"]) for row in applicable), Decimal(0))
        tax_paid = sum((decimal(row["tax_payment"]) for row in applicable), Decimal(0))
        tax_balance = tax_accrual - tax_paid
        if tax_balance >= 0:
            tax_payable, tax_prepaid = tax_balance, Decimal(0)
        else:
            tax_payable, tax_prepaid = Decimal(0), -tax_balance
        flow_totals = {kind: sum((decimal(flow["amount"]) for flow in plan["flows"] if flow["flow_type"] == kind and flow["flow_date"] <= text), Decimal(0)) for kind in ("capital_expenditure", "owner_contribution", "owner_distribution")}
        financing = sum((decimal(row["amount"]) for row in operations if row["event_type"] == "融资" and row["event_date"] <= text), Decimal(0))
        bonds = sum((decimal(row["amount"]) for row in operations if row["event_type"] == "债券发行" and row["event_date"] <= text), Decimal(0))
        loan_balance = Decimal(0)
        for ledger in ledgers:
            checkpoints = [row for row in ledger["checkpoints"] if row["balance_as_of_date"] <= text]
            if checkpoints:
                loan_balance += decimal(checkpoints[-1]["outstanding_principal"])
        local = rng(config["seed"], "working-capital-v102", number, text)
        receivable_months = fraction(local, ".22", ".34") if basic["industry"] not in ("批发零售", "餐饮住宿") else fraction(local, ".06", ".14")
        payable_months = fraction(local, ".26", ".40")
        customer_advance_rate = fraction(local, ".035", ".085")
        supplier_credit = (sum((decimal(row["direct_cost"]) * Decimal(".09") for row in applicable), Decimal(0)) if "business_exit_and_asset_review" in features else Decimal(0))
        receivables = decimal(money(base_receivables + revenue * receivable_months))
        inventory = decimal(final_process["drivers"]["closing_inventory"])
        payables = decimal(money(base_payables + direct * payable_months + supplier_credit))
        operating_other = decimal(money(base_other + revenue * customer_advance_rate))
        capex, contribution, distribution = (flow_totals[key] for key in ("capital_expenditure", "owner_contribution", "owner_distribution"))
        fixed = base_fixed + capex - depreciation - impairment
        cash = (base_cash + retained + depreciation + impairment + tax_accrual - tax_paid - (receivables - base_receivables) - (inventory - base_inventory) - capex +
                (payables - base_payables) + (operating_other - base_other) + loan_balance + bonds + contribution + financing - distribution)
        if min(cash, receivables, inventory, fixed, payables, operating_other, loan_balance, bonds) < 0:
            details = {
                "cash": cash, "fixed_assets": fixed, "trade_receivables": receivables,
                "inventory": inventory, "trade_payables": payables, "operating_other_liabilities": operating_other,
                "bank_loan_principal": loan_balance, "bond_principal": bonds,
                "cumulative_net_profit": retained, "cumulative_income_tax_expense": tax_accrual,
                "cumulative_tax_paid": tax_paid, "cumulative_capital_expenditure": capex,
                "cumulative_depreciation": depreciation, "cumulative_asset_impairment": impairment,
                "cumulative_owner_contributions": contribution, "cumulative_equity_financing": financing,
                "cumulative_owner_distributions": distribution,
            }
            detail_text = ", ".join(f"{key}={money(value)}" for key, value in details.items())
            opening_text = ", ".join(f"{key}={value}" for key, value in opening.items() if key.startswith("opening_"))
            raise ValueError(f"{basic['enterprise_id']} {text}: preselected cash/asset plan becomes negative; "
                             f"{detail_text}; {opening_text}; change the business plan rather than clipping a balance")
        fixed_and_other = fixed + tax_prepaid
        total_assets = cash + receivables + inventory + fixed_and_other
        other = operating_other + bonds + tax_payable
        total_liabilities = loan_balance + payables + other
        equity = base_equity + retained + contribution + financing - distribution
        if total_assets - total_liabilities != equity:
            raise ValueError(f"{basic['enterprise_id']} {text}: named funding and operating rollforward differs from stock equity")
        row = row_factory("enterprise_operation_profile", profile_record_id=f"PR-{number:06d}-{text.replace('-', '')}-R1",
            enterprise_id=basic["enterprise_id"], as_of_date=text, has_wechat_account=source_profile["has_wechat_account"],
            wechat_content_summary=None, total_assets=money(total_assets), total_liabilities=money(total_liabilities), amount_unit="元")
        if row["has_wechat_account"]:
            row["wechat_content_summary"] = f"截至{text}，公开介绍{basic['industry']}产品服务、经营场所、岗位招聘及日常交付安排。"
            if "business_exit_and_asset_review" in features:
                row["wechat_content_summary"] += "退出非核心业务后保留最低场地和供应承诺，设备用途与可回收性定期复核。"
            if any(decimal(process["drivers"]["rework_units"]) > 0 for process in applicable):
                row["wechat_content_summary"] += ("部分商品批次按质量处置协议开展无偿替换及复检，逆向物流和处理服务按协议安排。" if basic["industry"] == "批发零售" else
                                                  "部分交付开展返工和复验，相关人员与外包产能按既定合同安排。")
        row.update(metadata(config, "enterprise_operation_profile", f"PROFILE-{basic['enterprise_id']}-{text}", text))
        versions = []
        if "lawful_revision" in features and text == "2025-12-31":
            # A lawful metadata/summary revision does not manufacture a second
            # economic stock. Numeric revision evidence is in the March flow.
            first = deepcopy(row)
            versions.append(first)
            row["profile_record_id"] = row["profile_record_id"].removesuffix("R1") + "R2"
            if row["has_wechat_account"]:
                row["wechat_content_summary"] += "已补充供应安排与交付复核资料。"
            row.update(metadata(config, "enterprise_operation_profile", row["source_record_ref"], text, 2, 12))
        versions.append(row)
        for version in versions:
            retained_change = Decimal(0) if previous_retained is None else retained - previous_retained
            owner_change = Decimal(0) if previous_owner is None else contribution - previous_owner
            financing_change = Decimal(0) if previous_financing is None else financing - previous_financing
            distribution_change = Decimal(0) if previous_distribution is None else distribution - previous_distribution
            result.append(version)
            audit.append({
                "enterprise_id": basic["enterprise_id"], "as_of_date": text, "report_revision": version["report_revision"],
                "record_ref": {"table": "enterprise_operation_profile", "primary_key": {"profile_record_id": version["profile_record_id"]}},
                **opening, "cash": money(cash), "trade_receivables": money(receivables), "inventory": money(inventory),
                "fixed_assets": money(fixed), "tax_prepaid": money(tax_prepaid), "tax_payable": money(tax_payable),
                "fixed_and_other_assets": money(fixed_and_other), "total_assets": money(total_assets), "bank_loan_principal": money(loan_balance),
                "bond_principal": money(bonds), "trade_payables": money(payables), "operating_other_liabilities": money(operating_other),
                "other_liabilities": money(other), "total_liabilities": money(total_liabilities), "owners_equity": money(equity),
                "retained_profit_change": money(retained_change), "owner_contributions_change": money(owner_change),
                "equity_financing_change": money(financing_change), "owner_distributions_change": money(distribution_change),
                "owner_and_other_equity_change": money(owner_change + financing_change - distribution_change),
                "cumulative_net_profit": money(retained), "cumulative_depreciation": money(depreciation), "cumulative_asset_impairment": money(impairment),
                "cumulative_income_tax_expense": money(tax_accrual), "cumulative_tax_paid": money(tax_paid),
                "cumulative_capital_expenditure": money(capex), "cumulative_owner_contributions": money(contribution),
                "cumulative_equity_financing": money(financing), "cumulative_owner_distributions": money(distribution),
                "funding_plan": deepcopy(plan),
                "vat_settlement_assumption": "same_period_pass_through_net_zero_no_vat_timing_assets_or_liabilities",
                "working_capital_drivers": {"latest_revenue": money(revenue), "latest_direct_cost": money(direct), "receivable_revenue_weight": receivable_months,
                    "inventory_process_mode": "monthly_preselected_buffer_rollforward",
                    "inventory_cost_weight": final_process["drivers"]["inventory_buffer_months"],
                    "inventory_capacity_budget": final_process["drivers"]["normal_monthly_input_budget"],
                    "inventory_target_factor": final_process["drivers"]["inventory_target_factor"],
                    "inventory_period_end": final_process["period_end"],
                    "payable_cost_weight": payable_months, "customer_advance_revenue_weight": customer_advance_rate,
                    "supplier_credit_extension": money(supplier_credit), "supplier_credit_cost_rate": "0.09" if "business_exit_and_asset_review" in features else "0",
                    "inventory_asset_kind": "unfinished_service_deliverables_and_unconsumed_inputs" if basic["industry"] in ("信息技术", "商务服务", "科研技术") else "unconsumed_materials_goods_and_work_in_process"},
                "mechanism": "窗口初始存量为原型估计；利润、预定股东投入及分配、真实借款及融资和债券、采购账期、客户预收、营运资金占用与设备更新逐项进入现金和权益桥。库存类资产按后续未耗用材料、未验收产品或服务项目在制投入规模安排，已费用化返工不再次资本化。折旧及减值只减少非现金资产；税费按权责计提、预缴按月实付，其差额按正负分别列应缴税负债或预缴税资产；各期本金取当时已发生的最新合同余额。所有者流量先定计划，不以最终权益差额倒算。",
            })
            previous_retained, previous_owner, previous_financing, previous_distribution = retained, contribution, financing, distribution
    return result, audit
