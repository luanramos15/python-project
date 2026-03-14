import os
import logging
import requests

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class ResponseService:
    """
    Service for generating automatic email responses using the Hugging Face
    Inference API (free tier).  Uses the OpenAI-compatible chat completions
    endpoint on router.huggingface.co.
    Falls back to smart templates when the API is unavailable.
    """

    TEMPLATES = {
        'Produtivo': (
            "Prezado(a),\n\n"
            "Agradecemos o seu contato referente a \"{subject}\".\n"
            "Recebemos sua solicitação e ela está sendo analisada pela nossa equipe. "
            "Retornaremos com mais informações o mais breve possível.\n\n"
            "Atenciosamente,\n"
            "Equipe de Atendimento"
        ),
        'Improdutivo': (
            "Prezado(a),\n\n"
            "Agradecemos a sua mensagem sobre \"{subject}\".\n"
            "Ficamos felizes com o seu contato.\n\n"
            "Atenciosamente,\n"
            "Equipe de Atendimento"
        ),
    }

    def __init__(self):
        self.api_token = os.getenv('HUGGINGFACE_API_TOKEN', '')
        self.model_name = os.getenv('HUGGINGFACE_MODEL_RESPONSE', DEFAULT_MODEL)
        self.api_url = HF_API_URL

        if not self.api_token:
            logger.warning(
                "HUGGINGFACE_API_TOKEN not set — response generation will use "
                "template fallback. Get a free token at "
                "https://huggingface.co/settings/tokens"
            )

        logger.info(f"ResponseService ready (model={self.model_name}, "
                     f"api_token={'set' if self.api_token else 'NOT SET'})")

    def gerar_resposta(self, content, subject="", category="Produtivo"):
        """
        Generate an automatic response for an email.

        1. Tries the HF Inference API with a chat-completions call.
        2. Falls back to a template if the API is unavailable or token is missing.
        """
        if self.api_token:
            try:
                ai_response = self._call_hf_api(content, subject, category)
                if ai_response:
                    return {
                        'response_text': ai_response,
                        'model_used': self.model_name,
                        'subtype': 'ai_generated',
                        'personalization_level': 'high',
                    }
            except Exception as e:
                logger.error(f"HF API call failed: {e}")

        # Fallback
        logger.info("Using template fallback for response generation")
        return self._template_response(subject, category)

    def _call_hf_api(self, content, subject, category):
        """Call the HF Inference API (chat completions) and return generated text."""
        system_msg, user_msg = self._build_messages(content, subject, category)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 400,
            "temperature": 0.7,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code == 503:
            logger.warning("Model is loading on HF servers, using fallback")
            return None

        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible response format
        choices = data.get('choices', [])
        if choices:
            generated = choices[0].get('message', {}).get('content', '').strip()
            if len(generated) > 20:
                return generated

        return None

    def _build_messages(self, content, subject, category):
        """Build system and user messages for the chat completions API."""
        truncated = content[:1500]

        system_msg = (
            "Você é um assistente profissional de atendimento ao cliente de uma "
            "empresa do setor financeiro. Sua função é redigir respostas de email "
            "em português brasileiro de forma educada e profissional. "
            "Escreva APENAS a resposta do email, sem explicações adicionais. "
            "Assine sempre como 'Equipe de Atendimento'."
        )

        if category == "Produtivo":
            user_msg = (
                f"Escreva uma resposta profissional para o email abaixo. "
                f"Reconheça a solicitação específica do cliente e informe que a "
                f"equipe está trabalhando nisso.\n\n"
                f"Assunto: {subject or '(sem assunto)'}\n"
                f"Email:\n{truncated}"
            )
        else:
            user_msg = (
                f"Escreva uma resposta curta e simpática para o email informal abaixo. "
                f"Agradeça a mensagem de forma calorosa e breve.\n\n"
                f"Assunto: {subject or '(sem assunto)'}\n"
                f"Email:\n{truncated}"
            )

        return system_msg, user_msg

    def _template_response(self, subject, category):
        """Generate a template-based response as fallback."""
        template = self.TEMPLATES.get(category, self.TEMPLATES['Produtivo'])
        text = template.format(subject=subject or "sua mensagem")
        return {
            'response_text': text,
            'model_used': 'template-fallback',
            'subtype': 'template',
            'personalization_level': 'medium',
        }


# Singleton — no model download, just config
response_service = ResponseService()
