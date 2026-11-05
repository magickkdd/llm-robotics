"""BudgetTransport: all HTTP is mocked, no key or .env needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import httpx
from inspect_robots_libero import budget
from inspect_robots_libero.budget import BudgetStop, BudgetTransport

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def reply(request):
            self.sent.append(json.loads(request.content))
            return httpx.Response(200,headers={'x-request-id':'req_mock'},json={
                'id':'resp_mock','usage':{'input_tokens':100,'output_tokens':50,
                'input_tokens_details':{'cached_tokens':20},
                'output_tokens_details':{'reasoning_tokens':30}},'output':[]})
        self.transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),model='openai/gpt-6-astra')
        self.client=httpx.Client(transport=self.transport)
    def tearDown(self):
        self.client.close(); self.transport.dispose(); self.tmp.cleanup()
    def request(self):
        return self.client.post('https://api.openai.com/v1/responses',json={
            'model':'gpt-6-astra','input':'offline','tools':[]})
    def test_requires_episode(self):
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_budget_blocks_before_send(self):
        self.transport.begin_episode(0); self.transport.total_budget=.001
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_accounting_caps_and_no_headers_logged(self):
        self.transport.begin_episode(0); self.request()
        self.assertEqual(self.sent[0]['max_output_tokens'],8192)
        self.assertEqual(self.sent[0]['service_tier'],'default')
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['request_id'],'req_mock')
        self.assertAlmostEqual(rows[-1]['estimated_usd'],.00332)
        self.assertNotIn('Authorization',json.dumps(rows))
    def test_full_output_cap_reserved_before_send(self):
        self.transport.begin_episode(0)
        self.transport.total_budget=.10
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent, [])

    def test_configurable_output_cap(self):
        self.transport.max_output_tokens=8192
        self.transport.begin_episode(0)
        self.request()
        self.assertEqual(self.sent[0]['max_output_tokens'],8192)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(rows[0]['reservation_usd'],8192*50/1e6)

    def test_missing_output_usage_keeps_reservation_and_stops(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(200,json={'usage':{'input_tokens':100}}))
        self.request()
        self.assertTrue(self.transport.uncertain)
        self.assertGreater(self.transport.used,0)
        with self.assertRaises(BudgetStop): self.request()

    def test_attempt_cap(self):
        self.transport.begin_episode(0); self.transport.calls=80
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_unknown_usage_stops_subsequent_calls(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(200,json={'output':[]}))
        self.request()
        with self.assertRaises(BudgetStop): self.request()
        self.assertTrue(self.transport.uncertain)

    def test_network_error_records_exception_type_without_request_headers(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ReadTimeout('mock timeout')))
        with (mock.patch.object(budget,'_READ_ERROR_BACKOFF_S',(0,0,0)),
              mock.patch.object(budget.time,'sleep')):
            with self.assertRaises(BudgetStop):
                self.request()
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        unknown=next(row for row in rows if row['event']=='usage_unknown')
        self.assertEqual(unknown['error_type'],'ReadTimeout')
        retries=[row for row in rows if row['event']=='read_error_retry']
        self.assertEqual(len(retries),3)
        self.assertEqual(unknown['reserved_usd'],4*retries[0]['additional_reservation_usd'])
        self.assertNotIn('Authorization',json.dumps(rows))

    def test_read_timeout_retries_then_succeeds_with_extra_reservation(self):
        self.transport.begin_episode(0)
        attempts={'count':0}
        def reply(request):
            attempts['count']+=1
            if attempts['count']==1:
                raise httpx.ReadTimeout('mock timeout')
            return httpx.Response(200,json={'id':'resp_mock','output':[],
                'usage':{'input_tokens':100,'output_tokens':50}})
        self.transport.inner=httpx.MockTransport(reply)
        with (mock.patch.object(budget,'_READ_ERROR_BACKOFF_S',(0,)),
              mock.patch.object(budget.time,'sleep')):
            self.assertEqual(self.request().status_code,200)
        self.assertEqual(attempts['count'],2)
        self.assertEqual(self.transport.calls,1)
        self.assertFalse(self.transport.uncertain)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['event']=='read_error_retry' for row in rows),1)
        self.assertGreater(self.transport.used,rows[0]['reservation_usd'])

    def test_connect_timeout_retries_then_succeeds_as_one_call(self):
        self.transport.begin_episode(0)
        attempts={'count':0}
        def reply(request):
            attempts['count']+=1
            if attempts['count']<3:
                raise httpx.ConnectTimeout('mock connect timeout')
            return httpx.Response(200,json={'id':'resp_mock','output':[],
                'usage':{'input_tokens':100,'output_tokens':50}})
        self.transport.inner=httpx.MockTransport(reply)
        with (mock.patch.object(budget,'_CONNECT_TIMEOUT_BACKOFF_S',(0,0)),
              mock.patch.object(budget.time,'sleep')):
            self.assertEqual(self.request().status_code,200)
        self.assertEqual(attempts['count'],3)
        self.assertEqual(self.transport.calls,1)
        self.assertFalse(self.transport.uncertain)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['event']=='connect_timeout_retry' for row in rows),2)

    def test_exhausted_connect_timeouts_release_reservation(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout('mock connect timeout')))
        with (mock.patch.object(budget,'_CONNECT_TIMEOUT_BACKOFF_S',(0,)),
              mock.patch.object(budget.time,'sleep')):
            with self.assertRaises(BudgetStop):
                self.request()
        self.assertFalse(self.transport.uncertain)
        self.assertEqual(self.transport.blocked,'connect_timeout_exhausted')
        self.assertAlmostEqual(self.transport.used,0)
        self.assertAlmostEqual(self.transport.episode_used,0)

    def test_server_error_retried_as_one_call(self):
        self.transport.begin_episode(0)
        replies=iter([httpx.Response(503,json={'error':'mock'}),httpx.Response(529,json={'error':'mock'})])
        inner=self.transport.inner
        self.transport.inner=httpx.MockTransport(
            lambda r:next(replies,None) or inner.handle_request(r))
        with mock.patch.object(budget,'_SERVER_ERROR_BACKOFF_S',(0,0,0)):
            self.assertEqual(self.request().status_code,200)
        self.assertEqual(self.transport.calls,1)
        self.assertFalse(self.transport.uncertain)
    def test_exhausted_server_error_releases_reservation(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(503,json={'error':'mock'}))
        with mock.patch.object(budget,'_SERVER_ERROR_BACKOFF_S',(0,)):
            self.assertEqual(self.request().status_code,503)
            self.assertEqual(self.request().status_code,503)
        self.assertFalse(self.transport.uncertain)
        self.assertAlmostEqual(self.transport.used,0)
    def test_forbid_other_models(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://api.openai.com/v1/responses',json={'model':'gpt-6-sol','input':'x'})
        self.assertEqual(self.sent,[])
    def test_unpriced_model_rejected(self):
        with self.assertRaises(ValueError):
            BudgetTransport(self.tmp.name,15,model='openai/some-unpriced-model')
    def test_forbid_other_hosts(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://example.com/v1/responses',json={'model':'gpt-6-astra'})
        self.assertEqual(self.sent,[])

    def test_spacetime_responses_endpoint(self):
        self.transport.dispose()
        received=[]
        def reply(request):
            received.append(json.loads(request.content))
            return httpx.Response(200,headers={'x-request-id':'spacetime_mock'},json={
                'id':'resp_mock','status':'completed','output':[],
                'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}})
        transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
            model='openai/gpt-6-luna',base_url='https://spacetimeai.cc')
        client=httpx.Client(transport=transport)
        try:
            self.assertEqual(transport.wire,'responses')
            self.assertEqual(transport.expected_path,'/v1/responses')
            transport.begin_episode(0)
            response=client.post('https://spacetimeai.cc/v1/responses',json={
                'model':'gpt-6-luna','input':'hello','tools':[]})
            self.assertEqual(response.status_code,200)
            self.assertEqual(received[0]['max_output_tokens'],8192)
            self.assertEqual(transport.uncertain,False)
        finally:
            client.close()
            transport.dispose()

    def test_spacetime_terra_responses_endpoint(self):
        received=[]
        def reply(request):
            received.append(json.loads(request.content))
            return httpx.Response(200,headers={'x-request-id':'terra_mock'},json={
                'id':'resp_terra_mock','status':'completed','output':[],
                'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}})
        transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
            model='openai/gpt-6.1-terra',base_url='https://spacetimeai.cc',
            endpoint_base_url='https://spacetimeai.cc')
        client=httpx.Client(transport=transport)
        try:
            self.assertEqual(transport.wire,'responses')
            self.assertEqual(transport.expected_path,'/responses')
            transport.begin_episode(0)
            response=client.post('https://spacetimeai.cc/responses',json={
                'model':'gpt-6.1-terra','input':'hello','tools':[]})
            self.assertEqual(response.status_code,200)
            self.assertEqual(received[0]['model'],'gpt-6.1-terra')
            self.assertFalse(transport.uncertain)
        finally:
            client.close()
            transport.dispose()

    def test_spacetime_root_responses_endpoint(self):
        received=[]
        def reply(request):
            received.append(json.loads(request.content))
            return httpx.Response(200,headers={'x-request-id':'spacetime_root_mock'},json={
                'id':'resp_mock','status':'completed','output':[],
                'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}})
        transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
            model='openai/gpt-6-luna',base_url='https://spacetimeai.cc',
            endpoint_base_url='https://spacetimeai.cc')
        client=httpx.Client(transport=transport)
        try:
            self.assertEqual(transport.expected_path,'/responses')
            transport.begin_episode(0)
            response=client.post('https://spacetimeai.cc/responses',json={
                'model':'gpt-6-luna','input':'hello','tools':[]})
            self.assertEqual(response.status_code,200)
            self.assertEqual(received[0]['max_output_tokens'],8192)
            self.assertFalse(transport.uncertain)
        finally:
            client.close()
            transport.dispose()

    def test_custom_responses_base_url_is_https_and_v1(self):
        self.assertEqual(budget.wire_for('openai/deepseek-v4.1-flash-1017',
                                         base_url='https://discovery-api.intern-ai.org.cn/v1'),
                         'responses')
        self.assertEqual(budget.wire_for('openai/gpt-6.1-sol',
                                         base_url='https://api.diosirapi.co.uk/v1'),
                         'responses')
        for base_url in ('http://discovery-api.intern-ai.org.cn/v1',
                         'https://example.com/v2','https://discovery-api.intern-ai.org.cn/v1/other'):
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                budget.wire_for('openai/deepseek-v4.1-flash-1017',base_url=base_url)

    def test_sol61_diosir_endpoint_is_pinned_to_responses_path(self):
        transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(
            lambda request:httpx.Response(200,json={'usage':{'input_tokens':100,'output_tokens':50},
                                                    'output':[]})),
            model='openai/gpt-6.1-sol',base_url='https://api.diosirapi.co.uk/v1')
        try:
            self.assertEqual(transport.expected_host,'api.diosirapi.co.uk')
            self.assertEqual(transport.expected_path,'/v1/responses')
            transport.begin_episode(0)
            with httpx.Client(transport=transport) as client:
                response=client.post('https://api.diosirapi.co.uk/v1/responses',json={
                    'model':'gpt-6.1-sol','input':'offline','tools':[]})
            self.assertEqual(response.status_code,200)
        finally:
            transport.dispose()

class AnthropicBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def reply(request):
            self.sent.append((request.headers.get('anthropic-beta'),json.loads(request.content)))
            return httpx.Response(200,headers={'request-id':'req_mock'},json={
                'id':'msg_mock','type':'message','role':'assistant','content':[],
                'usage':{'input_tokens':100,'output_tokens':50,
                         'cache_creation_input_tokens':0,'cache_read_input_tokens':0}})
        self.transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
                                       model='anthropic/claude-opus-5-5')
        self.client=httpx.Client(transport=self.transport)
    def tearDown(self):
        self.client.close(); self.transport.dispose(); self.tmp.cleanup()
    def test_messages_accounting(self):
        self.transport.begin_episode(0)
        self.client.post('https://api.anthropic.com/v1/messages',json={
            'model':'claude-opus-5-5','max_tokens':1000,'messages':[],'tools':[]})
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertAlmostEqual(rows[-1]['estimated_usd'],(100*4.+50*20.)/1e6)
    def test_stale_thinking_blocks_dropped(self):
        self.transport.begin_episode(0)
        self.client.post('https://api.anthropic.com/v1/messages',headers={'anthropic-beta':'fast-mode-2026-02-01'},
                         json={'model':'claude-opus-5-5','max_tokens':1000,'messages':[],'tools':[],
                               'thinking':{'type':'adaptive'}})
        beta,body=self.sent[-1]
        self.assertEqual(body['thinking']['block_binding'],{'prefix_mismatch_behavior':'drop_block'})
        self.assertEqual(beta,'fast-mode-2026-02-01,'+budget._THINKING_BINDING_BETA)
    def test_thinking_signatures_counted_at_half_a_token_per_byte(self):
        self.transport.begin_episode(0)
        thinking={'type':'thinking','thinking':'','signature':'x'*700000}
        self.client.post('https://api.anthropic.com/v1/messages',json={
            'model':'claude-opus-5-5','max_tokens':1000,'tools':[],
            'messages':[{'role':'assistant','content':[thinking]}]})
        _,body=self.sent[-1]
        self.assertEqual(len(body['messages'][0]['content'][0]['signature']),700000)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertLess(rows[0]['input_token_reservation'],400000)
        self.assertGreater(rows[0]['input_token_reservation'],350000)
    def test_openai_endpoint_rejected(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://api.openai.com/v1/responses',json={'model':'claude-opus-5-5'})

class GeminiBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def reply(request):
            self.sent.append(json.loads(request.content))
            # completion_tokens leaves out thinking; total_tokens includes it.
            return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1000,'completion_tokens':50,'total_tokens':1350,
                         'prompt_tokens_details':{'cached_tokens':600}}})
        self.transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
                                       model='google/gemini-4-argon',max_output_tokens=8192)
        self.client=httpx.Client(transport=self.transport)
    def tearDown(self):
        self.client.close(); self.transport.dispose(); self.tmp.cleanup()
    def test_output_cap_and_thinking_billed(self):
        self.transport.begin_episode(0)
        self.client.post('https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',json={
            'model':'gemini-4-argon','messages':[],'tools':[]})
        self.assertEqual(self.sent[0]['max_completion_tokens'],8192)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        # 400 uncached input, 600 cached input, 350 output (50 visible + 300 thinking).
        self.assertAlmostEqual(rows[-1]['estimated_usd'],(400*2.+600*0.1+350*10.)/1e6)
    def test_other_google_endpoint_rejected(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://generativelanguage.googleapis.com/v1beta/interactions',json={'model':'gemini-4-argon'})

if __name__=='__main__': unittest.main()
