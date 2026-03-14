#!/usr/bin/env python3
"""
Simple test script to test the Email Classification API.
Run after starting the API with: python test_api.py
"""

import json
import sys
from typing import Dict, Any

try:
    import requests
except ImportError:
    print("⚠️  requests library not installed. Install with: pip install requests")
    sys.exit(1)


class APITester:
    """Test the Email Classification API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_health(self) -> bool:
        """Test the health check endpoint"""
        print("\n" + "="*60)
        print("TEST 1: Health Check")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"✓ Status: {response.status_code}")
            print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to API. Is it running on http://localhost:5000?")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_api_info(self) -> bool:
        """Test the API info endpoint"""
        print("\n" + "="*60)
        print("TEST 2: API Information")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.base_url}/")
            print(f"✓ Status: {response.status_code}")
            data = response.json()
            print(f"✓ Application: {data.get('application')}")
            print(f"✓ Version: {data.get('version')}")
            print(f"✓ Endpoints available: {len(data.get('endpoints', {}))}")
            return response.status_code == 200
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_productive_email(self) -> bool:
        """Test classification of a productive email"""
        print("\n" + "="*60)
        print("TEST 3: Classify Productive Email")
        print("="*60)
        
        payload = {
            "assunto": "Erro Crítico no Sistema de Pagamentos",
            "conteudo": "Prezados, identificamos um bug crítico no sistema de pagamentos em produção. "
                       "O módulo de processamento está fora do ar e afetando os clientes. "
                       "Precisamos de assistência imediata para resolver este problema. "
                       "Favor informar o prazo para correção."
        }
        
        print(f"Sending email:")
        print(f"  Subject: {payload['assunto']}")
        print(f"  Content: {payload['conteudo'][:50]}...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/emails/processar",
                json=payload
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 201:
                data = response.json()
                categoria = data['classificacao']['categoria']
                confianca = data['classificacao']['confianca']
                
                print(f"✓ Classification: {categoria}")
                print(f"✓ Confidence: {confianca:.2%}")
                print(f"✓ Email ID: {data['email_id']}")
                print(f"✓ Suggested response preview:")
                resposta = data['resposta_sugerida']['texto'][:100]
                print(f"   {resposta}...")
                
                # Expected: Produtivo
                if categoria == "Produtivo":
                    print("✓ Classification is correct!")
                    return True
                else:
                    print(f"⚠️  Expected 'Produtivo' but got '{categoria}'")
                    return False
            else:
                print(f"✗ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_unproductive_email(self) -> bool:
        """Test classification of an unproductive email"""
        print("\n" + "="*60)
        print("TEST 4: Classify Unproductive Email")
        print("="*60)
        
        payload = {
            "assunto": "Boas Festas!",
            "conteudo": "Desejamos a você e sua família um maravilhoso final de ano! "
                       "Que 2025 traga grandes oportunidades e muito sucesso para todos. "
                       "Aguardamos você no próximo ano com entusiasmo."
        }
        
        print(f"Sending email:")
        print(f"  Subject: {payload['assunto']}")
        print(f"  Content: {payload['conteudo'][:50]}...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/emails/processar",
                json=payload
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 201:
                data = response.json()
                categoria = data['classificacao']['categoria']
                confianca = data['classificacao']['confianca']
                
                print(f"✓ Classification: {categoria}")
                print(f"✓ Confidence: {confianca:.2%}")
                print(f"✓ Email ID: {data['email_id']}")
                print(f"✓ Suggested response preview:")
                resposta = data['resposta_sugerida']['texto'][:100]
                print(f"   {resposta}...")
                
                # Expected: Improdutivo
                if categoria == "Improdutivo":
                    print("✓ Classification is correct!")
                    return True
                else:
                    print(f"⚠️  Expected 'Improdutivo' but got '{categoria}'")
                    return False
            else:
                print(f"✗ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_list_emails(self) -> bool:
        """Test listing processed emails"""
        print("\n" + "="*60)
        print("TEST 5: List Processed Emails")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.base_url}/api/emails")
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                page = data.get('current_page', 1)
                pages = data.get('pages', 1)
                
                print(f"✓ Total emails: {total}")
                print(f"✓ Current page: {page}/{pages}")
                print(f"✓ Emails in response: {len(data.get('emails', []))}")
                
                if total > 0:
                    first_email = data['emails'][0]
                    print(f"✓ First email subject: {first_email.get('subject', 'N/A')}")
                
                return True
            else:
                print(f"✗ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def run_all_tests(self) -> None:
        """Run all tests and display summary"""
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  Email Classification API - Test Suite".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")
        
        results = {
            "Health Check": self.test_health(),
            "API Information": self.test_api_info(),
            "Productive Email": self.test_productive_email(),
            "Unproductive Email": self.test_unproductive_email(),
            "List Emails": self.test_list_emails(),
        }
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        print("")
        print(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✅ All tests passed! API is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
        
        print("\n" + "="*60)
        print("Next steps:")
        print("  - See README-BACKEND.md for full API documentation")
        print("  - Check ARCHITECTURE.md for system design details")
        print("  - View logs: docker compose logs -f app")
        print("="*60)


if __name__ == "__main__":
    print("Email Classification API - Test Suite")
    print("Starting tests...")
    
    tester = APITester()
    tester.run_all_tests()
