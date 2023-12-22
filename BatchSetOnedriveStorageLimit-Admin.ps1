#Set Parameters
$AdminSiteURL="https://YOURORG-admin.sharepoint.com"
 
#Get Credentials to connect
$Cred = Get-Credential
  
#Connect to Tenant Admin Site
Connect-PnPOnline $AdminSiteURL -Credentials $Cred
  
#Get All OneDrive for Business Sites
$OneDriveSites = Get-PnPTenantSite -IncludeOneDriveSites -Filter "Url -like '-my.sharepoint.com/personal/'"
 
#Loop through each site
ForEach($Site in $OneDriveSites)
{ 
    #Print current site,storage limit and used quota
    Write-host "Site URL:"$Site.Url -f Yellow
    Write-host "Storage Limit:"$Site.StorageQuota "MB" -f Yellow
    Write-host "Storage Used:"$Site.StorageUsageCurrent "MB" -f Yellow
    #Set storage limit to 5TB
    Set-SPOSite -Identity $Site.Url -StorageQuota 5242880
}