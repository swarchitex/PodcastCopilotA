from azure.identity import InteractiveBrowserCredential

from azure.keyvault.secrets import SecretClient

keyVaultName = "est-kv-copilot-podcast"
keyVaultEndpoint = f"https://{keyVaultName}.vault.azure.net"
allowed_tenants_to_pass:str ='c0551489-3ae7-4760-b774-7f08755c4158'

secret_bing_EndpointName = "est-kv-sc-bings-endpoint-copilot-podcast"
secret_bing_KeyName = "est-kv-sc-bings-key-copilot-podcast-1"

secret_openai_EndpointName = "est-kv-sc-openai-endpoint-copilot-podcast"
secret_openai_KeyName = "est-kv-sc-openai-key-copilot-podcast-1"
secret_openai_DeploymentName="est-kv-sc-openai-identifier-copilot-podcast-deployment-name"

secret_openai_dall_e_EndpointName = "est-kv-sc-openai-endpoint-copilot-podcast-dalle"
secret_openai_dall_e_KeyName = "est-kv-sc-openai-key-copilot-podcast-dalle-1"



AzureADCred = InteractiveBrowserCredential(additionally_allowed_tenants=allowed_tenants_to_pass)


def NullToEmptyString (valor:str|None) -> str :
    return "" if valor is None else valor


def GetSecret(keyName,endpoint:str) -> tuple[str, str] :
	client = SecretClient(vault_url=keyVaultEndpoint, credential=AzureADCred)

	k:str = NullToEmptyString(client.get_secret(keyName).value)
	e:str = NullToEmptyString(client.get_secret(endpoint).value)
	client.close()
	return k,e


def GetBingSearchSecrets() -> tuple[str, str] :
    return GetSecret(secret_bing_KeyName,secret_bing_EndpointName)


def GetAzureOpenAISecrets() -> tuple[str, str, str] :
	client = SecretClient(vault_url=keyVaultEndpoint, credential=AzureADCred)
	k:str = NullToEmptyString(client.get_secret(secret_openai_KeyName).value)
	e:str = NullToEmptyString(client.get_secret(secret_openai_EndpointName).value)
	n = NullToEmptyString(client.get_secret(secret_openai_DeploymentName).value)
	client.close()
	
	return k,e,n


def GetAzureOpenAI_Dall_E_Secrets() -> tuple[str, str] :
	return GetSecret(secret_openai_dall_e_KeyName,secret_openai_dall_e_EndpointName)
