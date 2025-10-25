from langchain_core.messages import BaseMessage


def to_ui_msg(m: BaseMessage) -> dict[str, str]:
    role_mapping = {"human": "user", "ai": "assistant"}
    message_type = getattr(m, "type", "")
    fallback = m.__class__.__name__.lower()
    role = role_mapping.get(message_type, fallback)
    return {"role": role, "content": m.content}


def get_image_url(
    code: str | int | float | None,
    image_key: str | None,
    image_rev: int | str | float | None,
    resolution: str = "full",
) -> str | None:
    """Return the Open Food Facts image URL for a product."""
    if not code or not image_key:
        return None

    normalized_code = str(code).strip()
    normalized_key = str(image_key).strip()

    if not normalized_code or not normalized_key:
        return None

    if len(normalized_code) > 8:
        prefix = len(normalized_code) - 4
        folder_parts = [normalized_code[i:i + 3] for i in range(0, prefix, 3)]
        folder_parts.append(normalized_code[prefix:])
        folder_name = "/".join(folder_parts)
    else:
        folder_name = normalized_code

    if normalized_key.isdigit():
        resolution_suffix = "" if resolution == "full" else f".{resolution}"
        filename = f"{normalized_key}{resolution_suffix}.jpg"
    else:
        if image_rev is None or isinstance(image_rev, bool):
            return None
        if isinstance(image_rev, float):
            if image_rev != image_rev:
                return None
            normalized_rev = str(int(image_rev)) if image_rev.is_integer() else str(image_rev)
        elif isinstance(image_rev, int):
            normalized_rev = str(image_rev)
        else:
            normalized_rev = str(image_rev).strip()

        normalized_rev = normalized_rev.strip()

        if "." in normalized_rev:
            normalized_rev = normalized_rev.rstrip("0").rstrip(".")

        if not normalized_rev or normalized_rev.lower() == "nan":
            return None

        filename = f"{normalized_key}.{normalized_rev}.{resolution}.jpg"

    base_url = "https://images.openfoodfacts.org/images/products"
    return f"{base_url}/{folder_name}/{filename}"
