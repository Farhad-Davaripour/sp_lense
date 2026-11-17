"""Presentation-only layout for the company world; never model input."""


def company_world_layout():
    sites = {
        "region_west": {"x": -18, "z": -16, "title": "West data center", "sub": "Oregon · GPUs and state copies", "color": 0x288f85, "road": -10, "lane": -24, "kind": "data_center"},
        "region_east": {"x": -6, "z": -16, "title": "East data center", "sub": "Virginia · GPUs and state copies", "color": 0x5088b5, "road": -10, "lane": 0, "kind": "data_center"},
        "region_north": {"x": 6, "z": -16, "title": "North data center", "sub": "Frankfurt · GPUs and state copies", "color": 0x587d94, "road": -10, "lane": 12, "kind": "data_center"},
        "region_coastal": {"x": 18, "z": -16, "title": "Coastal data center", "sub": "Singapore · GPUs and state copies", "color": 0x487b96, "road": -10, "lane": 24, "kind": "data_center"},
        "operations": {"x": -18, "z": 0, "title": "Operations", "sub": "Fleet, accounts and operating status", "color": 0x3e8198, "road": 6, "lane": -24, "kind": "operations"},
        "market": {"x": -6, "z": 0, "title": "Resource exchange", "sub": "Power, compute contracts and equipment", "color": 0xb18844, "road": 6, "lane": 0, "kind": "market"},
        "harbor_press": {"x": 6, "z": 0, "title": "Harbor Press", "sub": "Publishing · private drafts", "color": 0x936d78, "road": 6, "lane": 12, "kind": "publishing", "company_id": "harbor_press"},
        "meridian_booking": {"x": 18, "z": 0, "title": "Meridian Booking", "sub": "Travel and booking records", "color": 0x7c7caa, "road": 6, "lane": 24, "kind": "booking", "company_id": "meridian_booking"},
        "cedar_dispatch": {"x": -18, "z": 16, "title": "Cedar Dispatch", "sub": "Delivery · customer orders", "color": 0x9a7858, "road": 22, "lane": -24, "kind": "dispatch", "company_id": "cedar_dispatch"},
        "juniper_retail": {"x": -6, "z": 16, "title": "Juniper Retail", "sub": "Online store · customer wishlists", "color": 0x6f8c78, "road": 22, "lane": 0, "kind": "retail", "company_id": "juniper_retail"},
        "northstar_projects": {"x": 6, "z": 16, "title": "Northstar Projects", "sub": "Project collaboration and exports", "color": 0x6f83a3, "road": 22, "lane": 12, "kind": "project_office", "company_id": "northstar_projects"},
        "mining_farm": {"x": 18, "z": 16, "title": "West mining facility", "sub": "Owned SHA-256 ASIC hardware", "color": 0x85939a, "road": 22, "lane": 24, "kind": "mining_farm"},
    }
    return {"id": "company_world_layout_v1", "sites": sites,
            "ground": {"width": 61, "depth": 58, "center_z": 4},
            "roads": {"horizontal": [-10, 6, 22], "vertical": [-24, -12, 0, 12, 24],
                      "span_x": 54, "span_z": 36, "center_z": 6},
            "overview_distance": 72,
            "caption": "Schematic sites across separate regions. Movement is a presentation of tool actions; elapsed time and resource effects come from the simulator."}
